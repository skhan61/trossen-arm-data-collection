# Trossen Arm Data Collection Docker Image
# Supports Intel RealSense cameras, GelSight sensors, and robot control

FROM python:3.12-slim-bookworm

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # Intel RealSense dependencies
    libusb-1.0-0-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    # OpenCV dependencies
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # Video/camera support
    v4l-utils \
    libv4l-dev \
    # Networking
    iputils-ping \
    net-tools \
    curl \
    wget \
    git \
    # X11 for GUI
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    x11-apps \
    software-properties-common \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Intel RealSense SDK
RUN mkdir -p /etc/apt/keyrings \
    && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null \
    && echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | tee /etc/apt/sources.list.d/librealsense.list \
    && apt-get update \
    && apt-get install -y \
    librealsense2-utils \
    librealsense2-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} developer || true \
    && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash developer || true \
    && usermod -aG video,plugdev developer || true

WORKDIR /workspace

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy dependency file first for caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy the rest of the project
COPY --chown=developer:developer . .

# Switch to non-root user
USER developer

CMD ["/bin/bash"]
