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
ssh <your_pid>@tinkercliffs2.arc.vt.edu
```

You will be prompted for your VT password (the same one for canvas/hokiespa)

Once in, request an interactive session like this (we want to run everything in an interactive session to not consume resources)

```bash
interact -A cmda4864fall2025
```

Once in the interactive session, run 

```bash
module load Python
```

This will load the latest version of Python on the compute node.
