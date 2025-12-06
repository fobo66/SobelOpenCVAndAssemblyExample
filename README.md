# SobelOpenCVAndAssemblyExample

Edge detection using Sobel operator written in OpenCV and assembly intrinsics. macOS and Linux only.

## Setup

To launch the demo, you first need to set up the build environment.

### macOS

```bash
xcode-select --install # if you don't have Xcode installed already
brew install opencv sse2neon cmake ninja
```
### Ubuntu

```bash
sudo apt update && auso apt install -y libopencv-dev cmake ninja
```

### Build

```bash
mkdir build
cd build
cmake ../
cmake --build .
cd ../
```

### Launch

```bash
./build/main
```
