pip install -e .

# curl for dataset, download if not already downloaded
if [ -d "asl_dataset" ]; then
  echo "asl_dataset already downloaded"
else
  echo "asl_dataset not downloaded, downloading now"
  curl -L "https://drive.google.com/uc?export=download&id=1ttU_syFUrysneeMnvUatFJKL4fyA3nlM&confirm=t" > asl_dataset.zip
  unzip -q asl_dataset.zip
fi

if [ -d "checkpoints" ]; then
  echo "checkpoints already downloaded"
else
  echo "checkpoints not downloaded, downloading now"
  curl -L "https://drive.google.com/uc?export=download&id=1yMgugzRl2-DoqM6Kozw3DlWaAMqnk3jr&confirm=t" > checkpoints.zip
  unzip -q checkpoints.zip
fi

if [ -d "demo" ]; then
  echo "demo already downloaded"
else
  echo "demo not downloaded, downloading now"
  curl -L "https://drive.google.com/uc?export=download&id=10lawwTHSfxzbG3r3zIcLldmZwfKVv2yd&confirm=t" > demo.zip
  unzip -q demo.zip
fi