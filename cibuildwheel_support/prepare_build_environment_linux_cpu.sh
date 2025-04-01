# Configure pip to use PyTorch extra-index-url for CPU
mkdir -p $HOME/.config/pip
echo "[global]
extra-index-url = https://download.pytorch.org/whl/cpu" > $HOME/.config/pip/pip.conf
