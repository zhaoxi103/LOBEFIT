# LOBEFIT - LEO satellite broadcast ephemeris fitting open-source software based on automatic differentiation technique &#x20;

LOBEFIT allows users to **fit LEO satellite broadcast ephemeris parameters using precise LEO satellite orbits.** The software's source code is written in Python and can be effortlessly run in the console. The algorithm employed within this software utilizes the technique of automatic differentiation. Therefore, users can concentrate on the implementation of various broadcast ephemeris models, without the need for manually deriving complicated partial derivatives with respect to broadcast ephemeris parameters. Centimeter-level fitting accuracy can be achieved by selecting appropriate broadcast models for different LEO satellites using this software.&#x20;

## Features ⭐

- 🚀 **Effortless Setup**: The configuration is only upon the self-defined configuration file, which users can open via any text editor.

- 🌈 **Good Extensibility**: Users can add new broadcast ephemeris models in brdModel.py, and do not need to mannually derive the partial derivatives with respect to the parameters.

## 🔗 Quick start with the software

## How to Install 🚀

> install the latest Python version.
>
> The Python can be found in the following link: <https://www.python.org/downloads/>.&#x20;
>
> A Python version 3.9 or higher is recommended.
>
> After the successful installation of Python, the user should install several packages that are required for the execution of the optimization code. Installation of packages should be done in the up-to-date versions.
>
> os
>
> sys
>
> argparse
>
> Numpy
>
> JAX (link: [GitHub - google/jax: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more](https://github.com/google/jax).)

### Create Configuration File 🐳

> One should create a file of txt format for configuration at any path.
>
> For an example, one can create a file named "config.ini" at "/home/zhaox/code"
>
> Then, we should add the following contents to the file, for an example:
>
> \[config]&#x20;
>
> Start_Time \= 2023 5 20 0 00 0.000&#x20;
>
> Indicator \= 202&#x20;
>
> Input_File \= /home/zhaox/ephfit/code/orbdata/modified_grac1400.txt&#x20;
>
> Input_Type \= PV
>
> &#x20;Interval \= 1&#x20;
>
> Para_Num \= 17&#x20;
>
> Para_Case \= case1&#x20;
>
> Epoch_FitNum \= 901&#x20;
>
> URER \= 0.4557&#x20;
>
> URETN \= 0.6294&#x20;
>
> OutPath \= /home/zhaox/ephfit/code/output

The detailed description is as the following table.

| Start_Time   | the first epoch for fitting (GPS Time)-year month day hour minute second                                                                                                                                                                             |
| :----------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Indicator    | the abbreviated designation for LEO satellite within the input file specifying precise satellite orbits                                                                                                                                              |
| Input_File   | The complete path and the file name of the input precise satellite orbit file                                                                                                                                                                        |
| Input_Type   | the type of the input precise satellite orbit file‘PV’ denotes the orbit file encompasses both the position and velocity data of the LEO satellite, whereas 'P' signifies that the file exclusively contains the position data of the LEO satellite. |
| Interval     | the interval between neighboring epochs (the unit is second)                                                                                                                                                                                         |
| Para_Num     | the number of LEO satellite broadcast ephemeris parameters                                                                                                                                                                                           |
| Para_Case    | the case number in the supplementary material pertaining to a specific scheme                                                                                                                                                                        |
| Epoch_FitNum | the quantity of selected epochs utilized for the fitting process                                                                                                                                                                                     |
| URER         | URE weight factors in the radial direction for different orbital altitude                                                                                                                                                                            |
| URETN        | URE weight factors in the cross and along direction for different orbital altitude                                                                                                                                                                   |
| OutPath      | the complete path of the output files                                                                                                                                                                                                                |

### How to Run

- change directory to "src" and use this command:

  ```bash
  python brdEphFit.py -ini /home/zhaox/ephfit/code/config.ini
  ```

Here we assume that the path of the configuration file that we created is /home/zhaox/ephfit/code/config.ini.

Waiting for a moment, we can obtain the result.

### How to obtain the result

You can find the result at the OutPath that you defined in the configuration file. At OutPath, the estimated value of broadcast ephemeris parameters are in file "eph_para" ; the fitting error and UREs are in file "URE_file". These two files can be open using any text editor.

## License 📜

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details. 📄
