#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

help_function() {
    echo "Usage: $0 [OPTION]"
    echo
    echo "This script checks and installs Docker if not present."
    echo "then runs the NanoSAM model on port 80."
    echo
    echo "-h      Show this help message and exit."
}

install_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Docker not detected. Installing..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
    fi
}

IMAGE="r8.im/jimothyjohn/nanosam"

main() {
    if [[ "$#" -gt 0 && "$1" == "-h" ]]; then
        help_function
        exit 0
    fi

    install_docker

    docker run --gpus all -p 5000:5000 $IMAGE
}

main "$@"
