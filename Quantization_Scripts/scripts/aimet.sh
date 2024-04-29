export AIMET_VARIANT=torch_cpu
echo $AIMET_VARIANT
export release_tag=1.30.0
export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"
export wheel_file_suffix="cp38-cp38-linux_x86_64.whl"
python3 -m pip install ${download_url}/AimetCommon-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}
python3 -m pip install ${download_url}/AimetTorch-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix} -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install ${download_url}/Aimet-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}
source /usr/local/lib/python3.8/site-packages/aimet_common/bin/envsetup.sh
python3 test_imports.py