import re
import os
import sys
import shutil
from typing import List
import subprocess as sp
from distutils import spawn
from pathlib import Path
import difflib
import shutil

import requests
import numpy as np
from gotran import load_cell
from bs4 import BeautifulSoup

# pip install requests
# pip install gitpython
# pip install lxml
# pip install BeautifulSoup4
# pip install html5lib
from git import Repo

from cmodel import CModel

HERE = Path(__file__).absolute().parent

MODEL_ODE_DIR = HERE.joinpath("models").joinpath("ode")
MODEL_PY_DIR = HERE.joinpath("models").joinpath("python")
MODEL_C_DIR = HERE.joinpath("models").joinpath("c")
MODEL_CELLML_DIR = HERE.joinpath("models").joinpath("cellml")


class CellML:
    def __init__(self):
        self.models = {}
        for url in [
            "https://models.physiomeproject.org/workspace?b_start:int=0",
            "https://models.physiomeproject.org/workspace?b_start:int=500",
        ]:
            response = requests.get(url)
            html = BeautifulSoup(response.text, features="lxml")

            self.models.update(
                {
                    x.text: x.get("href")
                    for x in html.find_all(
                        "a",
                        attrs={"class": "contenttype-workspace state-published url"},
                    )
                }
            )

    def list_models(self, name=None):

        if name is not None:
            model_names = [
                model_name
                for model_name in self.models.keys()
                if name.lower() in model_name.lower()
            ]
        else:
            model_names = list(self.models.keys())

        return model_names

    def download_model(self, model_name, overwrite=False):

        model_names = self.list_models(model_name)
        msg = f"Cannot find model {model_name}, did you mean any of {model_names}"
        assert model_name in self.models, msg

        url = self.models[model_name]
        output = MODEL_CELLML_DIR.joinpath(Path(url).stem)
        if output.exists():
            if overwrite:
                shutil.rmtree(output)
            else:
                print("Model allready exist locally")
                return

        print(f"Download model {model_name} for {url} to {output}")
        Repo.clone_from(url, output)


def cpath(model):
    odefile = get_full_ode_path(model)
    return Path(re.sub(str(MODEL_ODE_DIR), str(MODEL_C_DIR), str(odefile))).with_suffix(
        ".h"
    )


def pypath(model):
    odefile = get_full_ode_path(model)
    return Path(
        re.sub(str(MODEL_ODE_DIR), str(MODEL_PY_DIR), str(odefile))
    ).with_suffix(".py")


def cbuild_dir(model):
    cfile = cpath(model)
    return MODEL_C_DIR.joinpath(f"build_{cfile.stem}")


def _list_models(directory, ext):
    return [model.name for model in directory.iterdir() if model.suffix == ext]


def list_ode_models():
    return _list_models(MODEL_ODE_DIR, ".ode")


def list_cellml_models():
    models = {d.stem: [] for d in MODEL_CELLML_DIR.iterdir() if d.is_dir()}

    for model in models:
        for f in MODEL_CELLML_DIR.joinpath(model).iterdir():
            if f.suffix == ".cellml":
                models[model].append(f.name)

    return models


def list_python_models():
    return _list_models(MODEL_PY_DIR, ".py")


def get_full_ode_path(model):

    for m in MODEL_ODE_DIR.iterdir():
        if model in str(m):
            return m
    all_models = ", ".join(list_ode_models())
    msg = f"Could not find model, available models are {all_models}"
    raise ValueError(msg)


def gotran2py(model):
    gotran2py = spawn.find_executable("gotran2py")
    odefile = get_full_ode_path(model)
    pyfile = pypath(model)
    sp.check_call([gotran2py, str(odefile), "--output", str(pyfile)])


def load_model(model, rebuild=False):

    # Check if model is present
    cfile = cpath(model)
    if not cfile.is_file() or rebuild:
        gotran2c(model)
        build_c(model)

    build_dir = cbuild_dir(model)
    lib = np.ctypeslib.load_library(next(build_dir.joinpath("lib").iterdir()), HERE)

    odefile = get_full_ode_path(model)
    ode = load_cell(str(odefile))
    return CModel(lib, ode)


def build_c(model):
    cfile = cpath(model)
    with open(MODEL_C_DIR.joinpath("template.c"), "r") as f:
        template = f.read()

    include_str = f'#include "{cfile.name}"\n'

    with open(MODEL_C_DIR.joinpath("demo.c"), "w") as f:
        f.write(include_str + template)

    model_name = cfile.stem

    build_dir = cbuild_dir(model)
    if build_dir.exists():
        shutil.rmtree(build_dir)

    build_dir.mkdir()

    sp.check_call(["cmake", f"-DCELL_LIBFILE={model_name}", ".."], cwd=build_dir)
    sp.check_call(["make"], cwd=build_dir)


def gotran2c(model):
    gotran2c = spawn.find_executable("gotran2c")

    odefile = get_full_ode_path(model)
    cfile = cpath(model)

    sp.check_call(
        [
            gotran2c,
            str(odefile),
            "--solvers.explicit_euler.generate=1",
            "--solvers.rush_larsen.generate=1",
            "--code.body.use_enum=1",
            "--output",
            str(cfile),
        ]
    )


def cellml2gotran(cellml_folder, cellml_file):
    cellml2gotran = spawn.find_executable("cellml2gotran")

    cellmlfile = MODEL_CELLML_DIR.joinpath(cellml_folder).joinpath(cellml_file)
    assert cellmlfile.is_file(), f"File {cellmlfile} does not exist"
    odepath = MODEL_ODE_DIR.joinpath(os.path.splitext(cellml_file)[0])

    sp.check_call([cellml2gotran, str(cellmlfile), "-o", str(odepath)])


if __name__ == "__main__":
    # print(list_ode_models())
    # gotran2c("tentusscher_model_2004_M")
    # load_model("tentusscher_model_2004_M", rebuild=True)
    fetch_cellml()
