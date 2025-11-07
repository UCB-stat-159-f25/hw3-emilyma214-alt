# Make or update the conda environment from environment.yml
env:
	conda env create -f environment.yml || conda env update -f environment.yml

# Build the MyST HTML site locally
html:
	myst build --html

# Clean generated output
clean:
	rm -rf figures/* audio/* _build/*
