

# Download data, set up folders
wget "https://mdumenci.s3.us-east-2.amazonaws.com/sr_20_batch740000_formulas606136.pickle"
mkdir checkpoint

# Set up environment
python3 -m venv venv && source venv/bin/activate
pip install torch pytorch-ignite numpy tqdm tensorboard future
git clone "https://github.com/mertdumenci/sat-ml.git" && cd "sat-ml"

mkdir runs
tensorboard --logdir runs &

# python branch_prediction.py --model lstm --data data/toy_batch0_formulas2556.pickle  --checkpoint checkpoint/ --batch-size 5 --log-interval 1