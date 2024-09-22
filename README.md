# LOBEFIT - LEO satellite broadcast ephemeris fitting open-source software based on automatic differentiation technique &#x20;

LOBEFIT allows users to **fit LEO satellite broadcast ephemeris parameters using precise LEO satellite orbits.** The software's source code is written in Python and can be effortlessly run both in the console and GUI. The algorithm employed within this software utilizes the technique of automatic differentiation. Therefore, users can concentrate on the implementation of various broadcast ephemeris models, without the need for manually deriving complicated partial derivatives with respect to broadcast ephemeris parameters. Centimeter-level fitting accuracy can be achieved by selecting appropriate broadcast models for different LEO satellites using this software. **==Please refer to the manual documentation in doc directory for detail.==**

## Features ⭐

- 🚀 **Effortless Setup**: In CUI (command-line user interface) mode, the configuration is only upon the self-defined configuration file, which users can open via any text editor; We also designed a simple UI using NiceGUI, a Python-based UI framework that works smoothly with the web browsers. Users may execute the GUI interface from the file located at /src/NiceGUIApp.py. The interface can be initiated by entering the command "python NiceGUIApp.py" in the console. The application is subsequently displayed within the default web browser environment on the local host.

- 🌈 **Good Extensibility**: Users can add new broadcast ephemeris models in brdModel.py, and do not need to mannually derive the partial derivatives with respect to the parameters.

## 🔗 Quick start with the software

## How to Install 🚀

> The Python can be found in the following link: <https://www.python.org/downloads/>.&#x20;
>
> A Python version 3.9 or higher is recommended.
>
> After the successful installation of Python, the user should install several packages that are required for the execution of the optimization code. Installation of packages should be done in the up-to-date versions.
>
> Numpy
>
> matplotlib
>
> NiceGUI (link: <https://nicegui.io/>)
>
> JAX (link: [GitHub - google/jax: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more](https://github.com/google/jax).)

### How to Run

- change directory to "src" and use this command:

  ```bash
  CUI mode:
  python brdEphFit.py -ini /home/zhaox/ephfit/code/config.ini
  GUI mode:
  python NiceGUIApp.py
  ```

Here we assume that the path of the configuration file that we created is /home/zhaox/ephfit/code/config.ini.

Waiting for a moment, we can obtain the result.&#x20;

### How to obtain the result

You can find the result at the OutPath that you defined in the configuration file. At OutPath, the estimated value of broadcast ephemeris parameters are in file "eph_para" ; the fitting error and UREs are in file "URE_file". These two files can be open using any text editor.

## License 📜

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details. 📄
