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

/readme_pictures/exemple.png

And here is the corresponding informations we need to extract

/readme_pictures/exemple_infos.png

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

