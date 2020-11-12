# Sphinx Documentation
Documentation for the Energy Intensity Indicators is built with [Sphinx](http://sphinx-doc.org/index.html). Here we provide a brief overview of how to work with the documentation.

## Installation [--VERIFY--]
Generating the docs requires the appropriate package:
```
conda install sphinx
conda install sphinx_rtd_theme

pip install ghp-import
pip install sphinx-click
```

## Building HTML Docs
### Mac/Linux
```
make html
```

### Windows
```
make.bat html
```

## Building PDF Docs
You'll need a LaTeX distribution:

### Mac/Linux
```
make latexpdf
```

### Windows
```
make.bat latexpdf
```

## Pushing to GitHub Pages
### Mac/Linux
```
make github
```

### Windows
```
make.bat html
```

Then run the git commands:
```
git branch -D gh-pages
git push origin --delete gh-pages
ghp-import -n -b gh-pages -m "Update documentation" ./_build/html
git checkout gh-pages
git push origin gh-pages
git checkout master # or whatever branch you were on
```
