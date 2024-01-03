# install dl_project package
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
  curl -L "https://drive.google.com/uc?export=download&id=1P3A-z5FcAo7i0memhHvOS7WUHT5NaDmo&confirm=t" > checkpoints.zip
  unzip -q checkpoints.zip
fi

if [ -d "generate" ]; then
  echo "generate already downloaded"
else
  echo "generate not downloaded, downloading now"
  curl -L "https://drive.google.com/uc?export=download&id=1mkbEAK7RJGrHLoS9tbu9OCbUStcTwRlr&confirm=t" > generate.zip
  unzip -q generate.zip
fi
