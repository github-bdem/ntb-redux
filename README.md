# ntb-redux

LM powered bot to play the video game nuclear throne. Currently in a very alpha state. There are tons of things that can be done to improve it and some of the design decisions (mainly the data cleaning layer) are questionable to say the least. This works as a decent proof of concept and a way to get all the steps for the project laid out. Next up I look forward to going through and improving/cleaning everything up by hand.

## History

This is a second attempt at a project I started around 7 years ago.  I had the basic idea for this project and even got around to creating the data collection portion [os-input-capture](https://github.com/github-bdem/os-input-capture) before life derailed my progess.  Fast forward to now, and with all the advances in the LM landscape and tooling, I figured it would be a good time to try out the project again.  This time I decided to also try out my new Anthropic subscription and see just how useful claude-code is when writing a project from the ground up.

## OS Dependencies

Currently only tested on Ubuntu 24 LTS. The following command will make sure all os level packages are installed.

- `sudo apt install scrot wmctrl xdotool xinput xev`

## From beginning to end:

### Collect Data

Ensure that nuclearthrone window is running and visible (preferably on the first level so we can skip menus tainting the dataset).

`npm run collect`

This will collect all keyboard and mouse input for 100 seconds along with screenshots of the target window and put them into a `training_data/session_TIMESTAMP` folder

### Clean the data

NOTE: This step is really whacky right now, it really really needs work.

Once you have the desired number of training data sets, we would want to clean those data sets and format them for our tensorflow inference layer training. Right now we have extremely rudimentary cleaning, but this is just a first up proof of concept.

`npm run clean-data`

Will clean data from `training_data` into `cleaned_data`.

### Train the model

Once we have the `cleaned_data` created we will want to finally train our model and save the trained weights.

`npm run train`

looks in `cleaned_data` and uses that info for training our tensorflow inference model, which it will then save in `models/model`

### Run the trained agent

We finally have a happy trained agent, time to allow it to play the game!

`npm run agent-mode`

will find the target window, by default `nuclearthrone`, starts grabbing screenshots of it, and then passes them into our trained tensorflow model (loaded from `models/model`).
