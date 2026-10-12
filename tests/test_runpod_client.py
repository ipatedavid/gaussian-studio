from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))

from runpod_client import PodCreationUncertainError, RunPodClient, RunPodError, RunPodCapacityError
from config import AppConfig
from job_controller import JobController


class PodCreationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.client = RunPodClient("test-key", "https://example.invalid/graphql")
        self.kwargs = dict(
            job_id="folder123", image_name="image", ssh_public_key="ssh-ed25519 key",
            gpu_type_id="gpu1", cloud_type="ALL", container_disk_gb=30,
            volume_gb=0, min_vcpu=1, min_memory_gb=8,
        )

    def test_existing_pod_prevents_mutation(self):
        self.client._graphql = Mock(return_value={"myself": {"pods": [
            {"id": "pod1", "name": "gaussian-folder123"}
        ]}})
        with self.assertRaisesRegex(RunPodError, "already exists"):
            self.client.create_pod(**self.kwargs)
        self.assertEqual(self.client._graphql.call_count, 1)

    def test_failed_preflight_prevents_mutation(self):
        self.client._graphql = Mock(side_effect=RunPodError("network down"))
        with self.assertRaises(RunPodError):
            self.client.create_pod(**self.kwargs)
        self.assertEqual(self.client._graphql.call_count, 1)

    def test_uncertain_creation_never_retries(self):
        def graphql(query, variables=None):
            if "mutation" in query:
                raise RunPodError("timeout")
            return {"myself": {"pods": []}}

        self.client._graphql = Mock(side_effect=graphql)
        with patch("runpod_client.time.sleep"):
            with self.assertRaisesRegex(PodCreationUncertainError, "No further creation attempt"):
                self.client.create_pod(**self.kwargs)
        mutations = [call for call in self.client._graphql.call_args_list
                     if "mutation" in call.args[0]]
        self.assertEqual(len(mutations), 1)

    def test_explicit_capacity_rejection_checks_pods_and_does_not_retry(self):
        def graphql(query, variables=None):
            if "mutation" in query:
                raise RunPodCapacityError("There are no longer any instances available")
            return {"myself": {"pods": []}}

        self.client._graphql = Mock(side_effect=graphql)
        with patch("runpod_client.time.sleep"):
            with self.assertRaisesRegex(RunPodCapacityError, "No capacity for the selected GPU"):
                self.client.create_pod(**self.kwargs)
        mutations = [call for call in self.client._graphql.call_args_list
                     if "mutation" in call.args[0]]
        self.assertEqual(len(mutations), 1)

    def test_capacity_rejection_with_mismatched_pod_requires_manual_inspection(self):
        def graphql(query, variables=None):
            if "mutation" in query:
                raise RunPodCapacityError("There are no longer any instances available")
            if self.client._graphql.call_count == 1:
                return {"myself": {"pods": []}}
            return {"myself": {"pods": [{
                "id": "pod1", "name": "gaussian-folder123", "imageName": "another-image",
            }]}}

        self.client._graphql = Mock(side_effect=graphql)
        with self.assertRaisesRegex(PodCreationUncertainError, "different imageName"):
            self.client.create_pod(**self.kwargs)
        mutations = [call for call in self.client._graphql.call_args_list
                     if "mutation" in call.args[0]]
        self.assertEqual(len(mutations), 1)

    def test_generic_error_recovers_verified_pod_after_listing_delay(self):
        request = None
        checks_after_mutation = 0

        def graphql(query, variables=None):
            nonlocal request, checks_after_mutation
            if "mutation" in query:
                request = variables["input"]
                raise RunPodError("Something went wrong. Please try again later")
            if request is None:
                return {"myself": {"pods": []}}
            checks_after_mutation += 1
            if checks_after_mutation == 1:
                return {"myself": {"pods": []}}
            matching = {key: request[key] for key in (
                "name", "imageName", "gpuCount", "containerDiskInGb",
                "volumeInGb", "ports", "dockerArgs",
            )}
            matching.update({
                "id": "pod1", "desiredStatus": "CREATED", "costPerHr": 0.49,
                "machine": {"gpuTypeId": "gpu1", "secureCloud": False},
            })
            return {"myself": {"pods": [matching]}}

        self.client._graphql = Mock(side_effect=graphql)
        with patch("runpod_client.time.sleep"):
            pod = self.client.create_pod(**self.kwargs)
        self.assertEqual(pod.pod_id, "pod1")
        self.assertTrue(pod.recovered_from_error)
        self.assertEqual(pod.cost_per_hour, 0.49)
        mutations = [call for call in self.client._graphql.call_args_list
                     if "mutation" in call.args[0]]
        self.assertEqual(len(mutations), 1)

    def test_success_requests_one_pod_and_ssh_port(self):
        self.client._graphql = Mock(side_effect=[
            {"myself": {"pods": []}},
            {"podFindAndDeployOnDemand": {"id": "pod1", "name": "gaussian-folder123"}}
        ])
        pod = self.client.create_pod(**self.kwargs)
        self.assertEqual(pod.pod_id, "pod1")
        self.assertEqual(self.client._graphql.call_count, 2)
        variables = self.client._graphql.call_args.args[1]
        self.assertEqual(variables["input"]["ports"], "22/tcp")
        self.assertEqual(variables["input"]["gpuCount"], 1)
        self.assertTrue(variables["input"]["supportPublicIp"])


class GpuAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.client = RunPodClient("test-key", "https://example.invalid/graphql")
        self.kwargs = dict(
            cloud_type="ALL", container_disk_gb=30, volume_gb=35,
            min_vcpu=1, min_memory_gb=8,
        )

    def test_query_matches_public_ip_and_required_disk(self):
        self.client._graphql = Mock(return_value={"gpuTypes": []})
        self.client.select_gpu_types(0.55, 24, **self.kwargs)
        variables = self.client._graphql.call_args.args[1]
        self.assertEqual(variables["community"]["minDisk"], 65)
        self.assertTrue(variables["community"]["supportPublicIp"])
        self.assertFalse(variables["community"]["secureCloud"])
        self.assertTrue(variables["secure"]["secureCloud"])

    def test_catalog_price_does_not_imply_capacity(self):
        self.client._graphql = Mock(return_value={"gpuTypes": [{
            "id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090",
            "memoryInGb": 24, "communityCloud": True, "secureCloud": True,
            "communityPrice": 0.34, "securePrice": 0.74,
            "communityOffer": {"stockStatus": None, "uninterruptablePrice": None},
            "secureOffer": {"stockStatus": "Low", "uninterruptablePrice": 0.74},
        }]})
        options = self.client.select_gpu_types(0.55, 24, **self.kwargs)
        self.assertEqual(len(options), 1)
        self.assertEqual(options[0]["_cloud_type"], "COMMUNITY")
        self.assertFalse(options[0]["_available"])
        self.assertEqual(options[0]["_stock_status"], "No verified SSH/disk capacity")

    def test_available_offer_uses_matching_tier_and_on_demand_price(self):
        self.client._graphql = Mock(return_value={"gpuTypes": [{
            "id": "gpu1", "displayName": "GPU 1", "memoryInGb": 24,
            "communityCloud": True, "secureCloud": False,
            "communityPrice": 0.22,
            "communityOffer": {"stockStatus": "Low", "uninterruptablePrice": 0.27},
        }]})
        options = self.client.select_gpu_types(0.55, 24, **self.kwargs)
        self.assertTrue(options[0]["_available"])
        self.assertEqual(options[0]["_hourly_price"], 0.27)
        self.assertEqual(options[0]["_cloud_type"], "COMMUNITY")
        self.assertEqual(options[0]["_price_kind"], "on-demand")


class FolderLaunchTests(unittest.TestCase):
    def test_job_thread_receives_photo_folder_and_selected_gpu(self):
        controller = JobController(AppConfig())
        photo_folder = Path("/tmp/photos")
        with patch("job_controller.threading.Thread") as thread:
            controller.run_job(photo_folder, ("selected-gpu", "COMMUNITY"))
        self.assertEqual(thread.call_args.kwargs["args"], (photo_folder, ("selected-gpu", "COMMUNITY")))
        thread.return_value.start.assert_called_once()

    def test_video_job_thread_receives_frame_target(self):
        controller = JobController(AppConfig())
        video = Path("/tmp/capture.mp4")
        with patch("job_controller.threading.Thread") as thread:
            controller.run_job(
                video, ("selected-gpu", "COMMUNITY"),
                source_type="video", video_fps=2.5,
            )
        self.assertEqual(thread.call_args.kwargs["kwargs"]["source_type"], "video")
        self.assertEqual(thread.call_args.kwargs["kwargs"]["video_fps"], 2.5)
        self.assertEqual(thread.call_args.kwargs["args"][0], video)


if __name__ == "__main__":
    unittest.main()
