from __future__ import annotations

import base64
import json
import time
from urllib.parse import quote, urljoin
from typing import Any, Optional

import requests


class RunPodError(RuntimeError):
    pass


class RunPodCapacityError(RunPodError):
    pass


class PodCreationUncertainError(RunPodError):
    pass


class RunPodClient:
    def __init__(self, api_key: str, graphql_url: str):
        if not api_key.strip():
            raise RunPodError("RunPod API key is empty.")

        self.api_key = api_key
        self.graphql_url = graphql_url.rstrip("/")
        self.rest_url = urljoin(self.graphql_url, "/v2/")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Low-level GraphQL
    # ------------------------------------------------------------------

    def _graphql(
        self,
        query: str,
        variables: Optional[dict] = None,
    ) -> dict:
        try:
            response = self.session.post(
                self.graphql_url,
                json={"query": query, "variables": variables or {}},
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RunPodError(
                f"RunPod API request failed: {exc}"
            ) from exc
        except ValueError as exc:
            raise RunPodError(
                "RunPod API returned invalid JSON."
            ) from exc

        data = payload.get("data")
        errors = payload.get("errors")

        if errors:
            message = "; ".join(
                str(error.get("message", error)) if isinstance(error, dict) else str(error)
                for error in errors
            )
            if "no longer any instances available" in message.lower():
                raise RunPodCapacityError(message)
            raise RunPodError(message)
        if not isinstance(data, dict):
            raise RunPodError("RunPod API returned no data.")
        return data

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # GPU catalog
    # ------------------------------------------------------------------

    def gpu_types(
        self, *, container_disk_gb: int, volume_gb: int,
        min_vcpu: int, min_memory_gb: int,
    ) -> list[dict]:
        # Match the deployment request. The unfiltered GPU type price is only a
        # catalog price; it says nothing about a public-IP host with enough disk.
        query = """
        query gpuTypes($community: GpuLowestPriceInput, $secure: GpuLowestPriceInput) {
          gpuTypes {
            id displayName memoryInGb communityCloud secureCloud
            communityPrice securePrice
            communityOffer: lowestPrice(input: $community) {
              stockStatus uninterruptablePrice
            }
            secureOffer: lowestPrice(input: $secure) {
              stockStatus uninterruptablePrice
            }
          }
        }
        """
        common = {
            "gpuCount": 1,
            "supportPublicIp": True,
            "minDisk": container_disk_gb + volume_gb,
            "minVcpuCount": min_vcpu,
            "minMemoryInGb": min_memory_gb,
        }
        variables = {
            "community": {**common, "secureCloud": False},
            "secure": {**common, "secureCloud": True},
        }
        return self._graphql(query, variables).get("gpuTypes", [])

    def select_gpu_types(
        self, max_price: float, min_vram_gb: int, *,
        cloud_type: str, container_disk_gb: int, volume_gb: int,
        min_vcpu: int, min_memory_gb: int,
    ) -> list[dict]:
        if cloud_type not in {"ALL", "COMMUNITY", "SECURE"}:
            raise RunPodError(f"Unsupported cloud type: {cloud_type}")
        candidates = []
        gpu_types = self.gpu_types(
            container_disk_gb=container_disk_gb, volume_gb=volume_gb,
            min_vcpu=min_vcpu, min_memory_gb=min_memory_gb,
        )
        for gpu in gpu_types:
            memory = self._to_float(gpu.get("memoryInGb")) or 0
            if memory < min_vram_gb:
                continue
            for tier, flag, list_key, offer_key in (
                ("COMMUNITY", "communityCloud", "communityPrice", "communityOffer"),
                ("SECURE", "secureCloud", "securePrice", "secureOffer"),
            ):
                if cloud_type != "ALL" and cloud_type != tier:
                    continue
                if not gpu.get(flag):
                    continue
                offer = gpu.get(offer_key) or {}
                on_demand_price = self._to_float(offer.get("uninterruptablePrice"))
                list_price = self._to_float(gpu.get(list_key))
                price = on_demand_price if on_demand_price and on_demand_price > 0 else list_price
                if price is None or price <= 0 or price > max_price:
                    continue
                stock = str(offer.get("stockStatus") or "").strip()
                available = stock.lower() in {"high", "medium", "low"} and on_demand_price is not None and on_demand_price > 0
                candidate = dict(gpu)
                candidate.update({
                    "_display_name": str(gpu.get("displayName") or gpu.get("id")),
                    "_cloud_type": tier,
                    "_hourly_price": price,
                    "_price_kind": "on-demand" if on_demand_price else "list",
                    "_stock_status": stock if available else "No verified SSH/disk capacity",
                    "_available": available,
                })
                candidates.append(candidate)
        rank = {"high": 0, "medium": 1, "low": 2}
        candidates.sort(key=lambda item: (
            not item["_available"],
            rank.get(item["_stock_status"].lower(), 3),
            item["_hourly_price"],
            item["_display_name"],
        ))
        return candidates

    # ------------------------------------------------------------------
    # Pod account state helpers
    # ------------------------------------------------------------------

    def list_pods(self) -> list[dict]:
        data = self._graphql("""query {
          myself { pods {
            id name desiredStatus imageName gpuCount
            containerDiskInGb volumeInGb ports dockerArgs
            costPerHr adjustedCostPerHr
            machine { gpuTypeId secureCloud }
          } }
        }""")
        myself = data.get("myself")
        if not isinstance(myself, dict) or not isinstance(myself.get("pods"), list):
            raise RunPodError("Could not verify existing pods; no pod will be created.")
        return myself["pods"]

    def create_pod(
        self, *, job_id: str, image_name: str,
        ssh_public_key: str, gpu_type_id: str,
        cloud_type: str, container_disk_gb: int, volume_gb: int,
        min_vcpu: int, min_memory_gb: int, engine: str = "nerfstudio",
    ):
        from models import PodInfo

        name = f"gaussian-{job_id}"
        existing = [p for p in self.list_pods() if p.get("name") == name]
        if existing:
            ids = ", ".join(str(p.get("id")) for p in existing)
            raise RunPodError(
                f"A pod named {name} already exists ({ids}). Inspect or terminate it in RunPod before launching again."
            )

        bootstrap = r'''# This script is the container's main process. It must NEVER exit: when it does,
# RunPod removes and recreates the container again and again, which looks from the
# outside like a flapping SSH port. Whatever goes wrong below, we end up in keepalive.
exec </dev/null
terminating=0
keepalive() {
    printf '[bootstrap] staying alive (last status %s)\n' "${1:-0}"
    # `sleep & wait` (not a plain sleep) so a stop signal is handled immediately.
    while true; do sleep 3600 & wait $!; done
}
trap 'rc=$?; [ "$terminating" = 1 ] || keepalive "$rc"' EXIT
trap 'terminating=1; exit 0' TERM INT
trap '' HUP PIPE

mkdir -p /root/.ssh /workspace/worker /workspace/job/input /workspace/job/output
chmod 700 /root/.ssh
printf '%s\n' "${SSH_PUBLIC_KEY:-}" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
need=""
command -v sshd >/dev/null 2>&1 || need="$need openssh-server"
command -v python3 >/dev/null 2>&1 || need="$need python3"
if [ -n "$need" ]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y $need || printf '[bootstrap] package install failed: %s\n' "$need"
fi
mkdir -p /run/sshd
# Some images (runpod/pytorch) ship sshd without host keys; it won't start without them.
ssh-keygen -A >/dev/null 2>&1 || printf '[bootstrap] ssh-keygen -A failed\n'

# One line of facts about this container, for diagnosing odd hosts.
mem=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo '?')
pids=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo '?')
printf '[bootstrap] container limits: cpus=%s memory=%s pids=%s\n' "$(nproc 2>/dev/null || echo '?')" "$mem" "$pids"

/usr/sbin/sshd -E /var/log/sshd.log || printf '[bootstrap] sshd failed to start\n'
sleep 1
printf '[bootstrap] ready for upload\n'

# Passive diagnostics only (no logins); failures here are irrelevant.
(
    listen=$(awk '$4=="0A" && $2 ~ /:0016$/ {print $2}' /proc/net/tcp /proc/net/tcp6 2>/dev/null | tr '\n' ' ')
    printf '[bootstrap] port 22 listeners (hex addr:port): %s\n' "${listen:-none}"
    ctr_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -n "$ctr_ip" ] && command -v ssh-keyscan >/dev/null 2>&1; then
        if timeout 12 ssh-keyscan -T 8 -p 22 "$ctr_ip" 2>/dev/null | grep -q ssh-; then
            printf '[bootstrap] sshd answers on container address %s\n' "$ctr_ip"
        else
            printf '[bootstrap] WARNING: sshd does NOT answer on container address %s\n' "$ctr_ip"
        fi
    fi
) || true
keepalive 0
'''
        bootstrap_b64 = base64.b64encode(bootstrap.encode()).decode("ascii")
        # The script must run from a FILE. Piping it into `bash` makes it the shell's
        # stdin, and any command that reads stdin (ssh, for one) silently eats the rest
        # of the script; the container's main process then exits and RunPod restarts it
        # in a loop.
        docker_args = (
            f"/bin/bash -lc 'echo {bootstrap_b64} | base64 -d > /tmp/gs-bootstrap.sh "
            "&& exec bash /tmp/gs-bootstrap.sh'"
        )
        environment = {
            "SSH_PUBLIC_KEY": ssh_public_key,
            "PYTHONUNBUFFERED": "1",
        }
        if engine == "spirula":
            # Spirula trains through Vulkan, which needs the NVIDIA graphics
            # driver libraries mounted into the container, not just compute.
            environment["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility,graphics"
            environment["NVIDIA_VISIBLE_DEVICES"] = "all"
        query = """
        mutation podFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput) {
          podFindAndDeployOnDemand(input: $input) {
            id name desiredStatus costPerHr adjustedCostPerHr gpus { id }
          }
        }
        """
        variables = {"input": {
            "cloudType": cloud_type,
            "gpuCount": 1,
            "gpuTypeId": gpu_type_id,
            "containerDiskInGb": container_disk_gb,
            "volumeInGb": volume_gb,
            "volumeMountPath": "/workspace",
            "minVcpuCount": min_vcpu,
            "minMemoryInGb": min_memory_gb,
            "imageName": image_name,
            "name": name,
            "ports": "22/tcp",
            "supportPublicIp": True,
            "dockerArgs": docker_args,
            "env": [{"key": k, "value": v} for k, v in environment.items()],
        }}
        try:
            data = self._graphql(query, variables)
        except RunPodError as exc:
            # A GraphQL error can arrive after the mutation created the pod.
            # Resolve the first request; never issue a second mutation here.
            found, check_error = self._find_created_pod(
                name=name, request=variables["input"],
            )
            if found is not None:
                return self._pod_info(found, gpu_type_id, recovered=True)
            if check_error:
                raise PodCreationUncertainError(
                    f"RunPod returned an error after pod creation was requested: {exc}. "
                    f"Could not safely verify the result ({check_error}). "
                    f"No further creation attempt will be made. Inspect RunPod for {name}; "
                    "a pod may be billing."
                ) from exc
            if isinstance(exc, RunPodCapacityError):
                raise RunPodCapacityError(
                    f"No capacity for the selected GPU ({gpu_type_id}). "
                    "Refresh the GPU list and choose another."
                ) from exc
            raise PodCreationUncertainError(
                f"RunPod returned an error after pod creation was requested: {exc}. "
                f"No matching pod appeared during verification. No further creation attempt "
                f"will be made. Inspect RunPod for {name} before trying again; "
                "a pod may still appear and bill."
            ) from exc
        pod = data.get("podFindAndDeployOnDemand")
        if not isinstance(pod, dict) or not pod.get("id"):
            found, check_error = self._find_created_pod(
                name=name, request=variables["input"],
            )
            if found is not None:
                return self._pod_info(found, gpu_type_id, recovered=True)
            detail = f" Verification problem: {check_error}." if check_error else ""
            raise PodCreationUncertainError(
                f"RunPod did not return a pod ID.{detail} Inspect RunPod for {name}; "
                "a pod may be billing. No further creation attempt will be made."
            )
        return self._pod_info(pod, gpu_type_id)

    def _find_created_pod(self, *, name: str, request: dict) -> tuple[dict | None, str | None]:
        last_error = None
        for attempt in range(10):
            if attempt:
                time.sleep(3)
            try:
                matches = [pod for pod in self.list_pods() if pod.get("name") == name]
            except RunPodError as exc:
                last_error = f"pod listing failed: {exc}"
                continue
            if len(matches) > 1:
                ids = ", ".join(str(pod.get("id")) for pod in matches)
                return None, f"multiple pods with this name exist: {ids}"
            if not matches:
                last_error = None
                continue
            pod = matches[0]
            expected = {
                "imageName": request["imageName"],
                "gpuCount": request["gpuCount"],
                "containerDiskInGb": request["containerDiskInGb"],
                "volumeInGb": request["volumeInGb"],
                "ports": request["ports"],
                "dockerArgs": request["dockerArgs"],
            }
            for field, value in expected.items():
                actual = pod.get(field)
                if actual is None:
                    last_error = f"pod {pod.get('id')} has not reported {field} yet"
                    break
                if actual != value:
                    return None, f"pod {pod.get('id')} has different {field}"
            else:
                if str(pod.get("desiredStatus") or "").upper() in {
                    "STOPPED", "EXITED", "TERMINATED"
                }:
                    return None, f"pod {pod.get('id')} is not starting"
                if (self._to_float(pod.get("adjustedCostPerHr")) is None
                        and self._to_float(pod.get("costPerHr")) is None):
                    last_error = f"pod {pod.get('id')} has not reported its price yet"
                    continue
                machine = pod.get("machine")
                if machine:
                    gpu_type = machine.get("gpuTypeId")
                    if gpu_type is not None and gpu_type != request["gpuTypeId"]:
                        return None, f"pod {pod.get('id')} has a different GPU"
                    secure = request["cloudType"] == "SECURE"
                    cloud = machine.get("secureCloud")
                    if cloud is not None and cloud != secure:
                        return None, f"pod {pod.get('id')} is on a different cloud tier"
                if not pod.get("id"):
                    return None, "matching pod has no ID"
                return pod, None
        return None, last_error

    def _pod_info(self, pod: dict, gpu_type_id: str, *, recovered: bool = False):
        from models import PodInfo

        adjusted = self._to_float(pod.get("adjustedCostPerHr"))
        normal = self._to_float(pod.get("costPerHr"))
        return PodInfo(
            pod_id=pod["id"], name=pod.get("name", ""),
            status=pod.get("desiredStatus", ""),
            desired_status=pod.get("desiredStatus", ""),
            gpu_name=gpu_type_id,
            cost_per_hour=adjusted if adjusted is not None else normal,
            recovered_from_error=recovered,
        )

    def pod_snapshot(self, pod_id: str) -> dict | None:
        """Read pod lifecycle and direct SSH details from the REST v2 API."""
        url = self.rest_url + "pods/" + quote(pod_id, safe="")
        try:
            response = self.session.get(url, timeout=20)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            pod = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RunPodError(f"Could not check pod {pod_id}: {exc}") from exc
        if not isinstance(pod, dict) or pod.get("id") != pod_id:
            raise RunPodError(f"RunPod returned invalid status for pod {pod_id}.")
        return pod

    @staticmethod
    def is_public_address(host: str) -> bool:
        """False for private/CGNAT/loopback addresses, which no outside client can reach."""
        import ipaddress
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return True  # a host name; assume it resolves publicly
        return not (
            address.is_private or address.is_loopback or address.is_link_local
            or address in ipaddress.ip_network("100.64.0.0/10")
        )

    @staticmethod
    def direct_ssh_endpoint(pod: dict) -> tuple[str, int] | None:
        direct = (pod.get("ssh") or {}).get("direct") or {}
        if direct.get("host") and direct.get("port"):
            return str(direct["host"]), int(direct["port"])
        for port in ((pod.get("runtime") or {}).get("ports") or []):
            if port.get("private") == 22 and port.get("ip") and port.get("public"):
                return str(port["ip"]), int(port["public"])
        return None

    def recent_system_logs(self, pod_id: str, seen_ids: set[str]) -> list[str]:
        """Fetch recent image/startup events without keeping an SSE stream open."""
        url = self.rest_url + "pods/" + quote(pod_id, safe="") + "/logs"
        lines = []
        event_id = None
        data_lines = []
        try:
            with self.session.get(
                url, params={"source": "system", "tail": 40},
                headers={"Accept": "text/event-stream"},
                stream=True, timeout=(5, 2),
            ) as response:
                if response.status_code == 404:
                    return []
                response.raise_for_status()
                for raw in response.iter_lines(chunk_size=1, decode_unicode=True):
                    line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
                    line = line.rstrip("\r")
                    if line.startswith("id:"):
                        event_id = line[3:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif not line:
                        if event_id and event_id not in seen_ids and data_lines:
                            try:
                                payload = json.loads("\n".join(data_lines))
                            except json.JSONDecodeError:
                                payload = None
                            if isinstance(payload, dict) and payload.get("source") == "system":
                                seen_ids.add(event_id)
                                message = str(payload.get("line") or "").strip()
                                if message:
                                    lines.append(message)
                        event_id = None
                        data_lines = []
                    if len(lines) >= 40:
                        break
        except requests.RequestException:
            # System logs are supplemental; pod status and heartbeat still work.
            pass
        return lines

    # ------------------------------------------------------------------
    # Pod inspection and control
    # ------------------------------------------------------------------

    def get_pod(self, pod_id: str) -> dict:
        query = """
        query pod($input: PodFilter) {
          pod(input: $input) {
            id name desiredStatus costPerHr adjustedCostPerHr
            runtime { ports { ip isIpPublic privatePort publicPort type } }
          }
        }
        """
        pod = self._graphql(query, {"input": {"podId": pod_id}}).get("pod")
        if not pod:
            raise RunPodError(f"Pod {pod_id} was not found.")
        return pod

    def terminate_pod(self, pod_id: str) -> None:
        query = """
        mutation podTerminate($input: PodTerminateInput!) {
          podTerminate(input: $input)
        }
        """
        self._graphql(query, {"input": {"podId": pod_id}})