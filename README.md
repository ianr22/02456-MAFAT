# 02456-MAFAT
COFGA dataset


## Commits
Commits cannot be made directly to master, but must instead be commited to secondary branches.

## Pull requests
When a pull request is made, the convention is as follows:  

**NAME:** Name the request after the implemented feature  
**Assignee:** Assign yourself as owner  
**Project:** Assign the pull request to a project if applicable  

**WARNING** If you delete your branch after a commit, you will have to reupload the pictures on first push!  
Instead, folow the instructions below
## How to delete a local branch and start a new one
This applies mostly when a pull request is merged, and you want to start over from master.

```bash
git checkout master
git branch -d <Branchname>
git checkout -b <Branchname>
git push --set-upstream origin <Branchname>
```
You now deleted your old branch, and created a new one and synced it to git


# CAPSTONE ARC INSTALL DIRECTIONS

## SETUP
1. Open terminal and run 

```bash
ssh <your_pid>@falcon1.arc.vt.edu
```

You will be prompted for your VT password (the same one for canvas/hokiespa)

Once in, request an interactive session like this (we want to run everything in an interactive session to not consume resources)

```bash
interact -A cmda4864fall2025 --gres=gpu:1 --mem=16G
```

This will allow enough emmory and resources for the model to run and modules to load.

Once in the interactive session, change directory to the ```02456-MAFAT``` folder and run 

```bash
source setup_scripts/load_modules.sh
```

This wil automatically load the needed modules to run each of the notebooks.

Once run, you need to start the jupyter server using 

```bash
jupyter lab --no-browser --ip=0.0.0.0 --port=8082
```

This will initiate the jupyter server that the notebooks will run on. In Visual Studio, 
hit the select kernel button on the top right of the notebook. Then a prompt will show up 
on the top search bar, and you will select existing jupyter server. copy this line of the output:
(under the text Or copy and paste one of these URLs)

```
http://fal<node>:8082/lab?token=<token_generated_by_server>
```

into the prompt and hit enter. This will connect the notebook to the server where the notebook can be run. 
