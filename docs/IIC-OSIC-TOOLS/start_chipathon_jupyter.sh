#!/bin/bash
# ========================================================================
# Start script for DIC docker images (use for Jupyter Notebooks only)
#
# SPDX-FileCopyrightText: 2022-2025 Harald Pretl and Georg Zachl
# Johannes Kepler University, Department for Integrated Circuits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ========================================================================

NB_STARTED=0
DOCKER_OK=0

HOST_USER="${SUDO_USER:-$USER}"
HOST_UID=$(id -u "${HOST_USER}" 2>/dev/null || id -u)
HOST_GID=$(id -g "${HOST_USER}" 2>/dev/null || id -g)
HOST_HOME=$(getent passwd "${HOST_USER}" 2>/dev/null | cut -d: -f6)
if [ -z "${HOST_HOME}" ]; then
	HOST_HOME="${HOME}"
fi

if [ -n "${DRY_RUN}" ]; then
	echo "[INFO] This is a dry run, all commands will be printed to the shell (Commands printed but not executed are marked with $)!"
	ECHO_IF_DRY_RUN="echo $"
fi

# SET YOUR DESIGN PATH RIGHT!
if [ -z ${DESIGNS+z} ]; then
	DESIGNS=$HOST_HOME/eda/designs
	if [ ! -d "$DESIGNS" ]; then
		${ECHO_IF_DRY_RUN} mkdir -p "$DESIGNS"
	fi
	[ -z "${IIC_OSIC_TOOLS_QUIET}" ] && echo "[INFO] Design directory auto-set to $DESIGNS."
fi

# Set the host ports, and disable them with 0. Only used if not set as shell variables!
if [ -z ${JUPYTER_PORT+z} ]; then
	JUPYTER_PORT=8888
fi
if [ -z ${DOCKER_USER+z} ]; then
	DOCKER_USER="hpretl"
fi

if [ -z ${DOCKER_IMAGE+z} ]; then
	DOCKER_IMAGE="iic-osic-tools"
fi

if [ -z ${DOCKER_TAG+z} ]; then
	DOCKER_TAG="chipathon"
fi

if [ -z ${CONTAINER_NAME+z} ]; then
	CONTAINER_NAME="iic-osic-tools_chipathon_jupyter_uid_${HOST_UID}"
fi

if [[ "$OSTYPE" == "linux"* ]]; then
	if [ -z ${CONTAINER_USER+z} ]; then
	        CONTAINER_USER=${HOST_UID}
	fi

	if [ -z ${CONTAINER_GROUP+z} ]; then
	        CONTAINER_GROUP=${HOST_GID}
	fi
else
	if [ -z ${CONTAINER_USER+z} ]; then
			CONTAINER_USER=1000
	fi

	if [ -z ${CONTAINER_GROUP+z} ]; then
			CONTAINER_GROUP=1000
	fi
fi

# Check for UIDs and GIDs below 1000, except 0 (root)
if [[ ${CONTAINER_USER} -ne 0 ]]  &&  [[ ${CONTAINER_USER} -lt 1000 ]]; then
        prt_str="# [WARNING] Selected User ID ${CONTAINER_USER} is below 1000. This ID might interfere with User-IDs inside the container and cause undefined behavior! #"
        printf -- '#%.0s' $(seq 1 ${#prt_str})
        echo
        echo "${prt_str}"
        printf -- '#%.0s' $(seq 1 ${#prt_str})
        echo
fi

if [[ ${CONTAINER_GROUP} -ne 0 ]]  && [[ ${CONTAINER_GROUP} -lt 1000 ]]; then
        prt_str="# [WARNING] Selected Group ID ${CONTAINER_GROUP} is below 1000. This ID might interfere with Group-IDs inside the container and cause undefined behavior! #"
        printf -- '#%.0s' $(seq 1 ${#prt_str})
        echo
        echo "${prt_str}"
        printf -- '#%.0s' $(seq 1 ${#prt_str})
        echo
fi

# Processing ports and other parameters
PARAMS="--security-opt seccomp=unconfined"
if [ "${JUPYTER_PORT}" -gt 0 ]; then
	PARAMS="$PARAMS -p $JUPYTER_PORT:8888"
fi

# On Linux servers, forward NVIDIA GPUs when available so local-HF inference can
# actually use CUDA inside the Chipathon container.
if [[ "$OSTYPE" == "linux"* ]]; then
	if command -v nvidia-smi >/dev/null 2>&1; then
		[ -z "${IIC_OSIC_TOOLS_QUIET}" ] && echo "[INFO] NVIDIA GPU detected, enabling Docker GPU passthrough."
		PARAMS="$PARAMS --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility"
	else
		[ -z "${IIC_OSIC_TOOLS_QUIET}" ] && echo "[INFO] No NVIDIA GPU detected on host."
	fi
fi

if [ -n "${DOCKER_EXTRA_PARAMS}" ]; then
	PARAMS="${PARAMS} ${DOCKER_EXTRA_PARAMS}"
fi

if [ -n "${IIC_OSIC_TOOLS_QUIET}" ]; then
	DOCKER_EXTRA_PARAMS="${DOCKER_EXTRA_PARAMS} -e IIC_OSIC_TOOLS_QUIET=1"
fi

if [ -n "${SUDO_USER}" ] && [ "${SUDO_USER}" != "root" ]; then
	[ -z "${IIC_OSIC_TOOLS_QUIET}" ] && echo "[INFO] Running under sudo; preserving host user context for ${SUDO_USER} (${HOST_UID}:${HOST_GID})."
fi

if ! docker info >/dev/null 2>&1; then
	echo "[ERROR] Docker daemon is not accessible for the current shell."
	echo "[HINT] Follow Docker's post-install steps to use Docker as a non-root user:"
	echo "       https://docs.docker.com/engine/install/linux-postinstall/"
	echo "[HINT] Typical fix:"
	echo "       sudo usermod -aG docker \$USER"
	echo "       newgrp docker"
	exit 1
fi
DOCKER_OK=1

# Check if the container exists and if it is running.
if [ "$(docker ps -q -f name="${CONTAINER_NAME}")" ]; then
	echo "[WARNING] Container is running!"
	echo "[HINT] It can also be stopped with \"docker stop ${CONTAINER_NAME}\" and removed with \"docker rm ${CONTAINER_NAME}\" if required."
	echo
	echo -n "Press \"s\" to stop, and \"r\" to stop & remove: "
	read -r -n 1 k <&1
	echo
	if [[ $k = s ]] ; then
		${ECHO_IF_DRY_RUN} docker stop "${CONTAINER_NAME}"
	elif [[ $k = r ]] ; then
		${ECHO_IF_DRY_RUN} docker stop "${CONTAINER_NAME}"
		${ECHO_IF_DRY_RUN} docker rm "${CONTAINER_NAME}"
	fi
# If the container exists but is exited, it is restarted.
elif [ "$(docker ps -aq -f name="${CONTAINER_NAME}")" ]; then
	echo "[WARNING] Container ${CONTAINER_NAME} exists."
	echo "[HINT] It can also be restarted with \"docker start ${CONTAINER_NAME}\" or removed with \"docker rm ${CONTAINER_NAME}\" if required."
	echo
	echo -n "Press \"s\" to start, and \"r\" to remove: "
	read -r -n 1 k <&1
	echo
	if [[ $k = s ]] ; then
		${ECHO_IF_DRY_RUN} docker start "${CONTAINER_NAME}"
		NB_STARTED=1
	elif [[ $k = r ]] ; then
		${ECHO_IF_DRY_RUN} docker rm "${CONTAINER_NAME}"
	fi
else
	[ -z "${IIC_OSIC_TOOLS_QUIET}" ] && echo "[INFO] Container does not exist, creating ${CONTAINER_NAME} ..."
	# Finally, run the container, and sets DISPLAY to the local display number
	${ECHO_IF_DRY_RUN} docker pull "${DOCKER_USER}/${DOCKER_IMAGE}:${DOCKER_TAG}" > /dev/null
	if [ -z "${ECHO_IF_DRY_RUN}" ] && [ $? -ne 0 ]; then
		echo "[ERROR] docker pull failed."
		exit 1
	fi
	# Disable SC2086, $PARAMS must be globbed and splitted.
	# shellcheck disable=SC2086
	${ECHO_IF_DRY_RUN} docker run -d --user "${CONTAINER_USER}:${CONTAINER_GROUP}" $PARAMS -v "$DESIGNS:/foss/designs:rw,z" --name "${CONTAINER_NAME}" "${DOCKER_USER}/${DOCKER_IMAGE}:${DOCKER_TAG}" -s /dockerstartup/scripts/run_GL.sh
	if [ -n "${ECHO_IF_DRY_RUN}" ] || [ $? -eq 0 ]; then
		NB_STARTED=1
	else
		echo "[ERROR] docker run failed."
		echo "[HINT] If the error mentions GPU capabilities, install and configure the NVIDIA Container Toolkit:"
		echo "       https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
		exit 1
	fi
fi

[ $NB_STARTED = 1 ] && [ $DOCKER_OK = 1 ] && echo "[INFO] Jupyter Notebook is running, point your browser to <http://localhost:8888>."
