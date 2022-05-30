# Project Ephesus

Interpret textual data generated from medical vocal memos

In the Library of Celsus in Ephesus, built in the 2nd century, there are four statues depicting wisdom (Sophia), knowledge (Episteme), intelligence (Ennoia) and excellence (Arete). Our project is named after this city and the goddess Sophia.

# What it's all about

After visiting a patient nurses and doctors need to quickly and easily send information

So they record a vocal memo after each visit

Today these memos are read by humans and the infos are manually entered in the database

We want to ease their work by automatically extracting informations from the vocal memos and pre-filling the informations to be entered in the database

# Dataset

4000 vocal memo recordings (4000 sentences)

14 targets to predict (up to 14 different pieces of informations per memo)

# Example

Here is an example of a memo

![memo example](/readme_pictures/exemple.png)

And here is the corresponding informations we need to extract

![memo infos example](/readme_pictures/exemple_infos.png)

# Our approach

## Preprocessing

Clean the data from stop words and punctuation

## Classification approach

Identify which part of the memo corresonds to which information

Have models convert each information into the target classes

## One-model-to-run-them-all approach

Have one model to interpret the whole of the memo

# Demo day

Make a demo prediction from a demo vocal memo

Show our success percentage

Give our feedback on possible improvement points and share the hypotheses we used to build our models

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for ephesus in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/ephesus`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "ephesus"
git remote add origin git@github.com:{group}/ephesus.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
ephesus-run
```

# Install

Go to `https://github.com/{group}/ephesus` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/ephesus.git
cd ephesus
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
ephesus-run
```
