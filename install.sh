mkdir PyAuto
cd PyAuto

conda_activate="/home/cao/anaconda3/bin/activate" #Set your own anaconda path here
source ${conda_activate}

# conda update -n base -c defaults conda
conda create -n PT_dev astropy scikit-image scikit-learn scipy python=3.10
conda activate PT_dev
pip install --upgrade pip

git clone https://github.com/rhayes777/PyAutoFit
git clone https://github.com/Jammy2211/PyAutoArray
git clone https://github.com/Jammy2211/PyAutoGalaxy
git clone https://github.com/Jammy2211/PyAutoLens

# switch branch
repositories=("PyAutoFit" "PyAutoArray" "PyAutoGalaxy" "PyAutoLens")
branch_name="2024.1.27.4"

for repo in "${repositories[@]}"
do
    echo "Switching to branch $branch_name in $repo"
    cd "$repo"
    git checkout "$branch_name"
    
    echo "install the requirement of $branch_name in $repo"
    pip install -r requirements.txt
    cd -
done

pip install autoconf

conda develop PyAutoFit
conda develop PyAutoArray
conda develop PyAutoGalaxy
conda develop PyAutoLens

pip install numba

#my package
pip install --upgrade GPy
pip install numba-scipy
pip install powerbox

#solve clash
pip install -U scikit-image==0.19.3

cd ..
git clone https://github.com/caoxiaoyue/lensing_potential_correction
conda develop lensing_potential_correction
