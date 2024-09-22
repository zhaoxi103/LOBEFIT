import numpy as np
from jax import config
from matplotlib import pyplot as plt
from nicegui import ui, run
import os
import socket

from brdEphFit import opttype
from brdEphFit import readConfig, readsp3File, ephfit, outputResult


class GenConfig:
    """
    After the input of the config parameters,
    The config file "config.ini" will be generated.
    """

    def __init__(self):
        configKey = ['Start_Time',
                     'Indicator',
                     'Input_File',
                     'Input_Type',
                     'Interval',
                     'Para_Num',
                     'Para_Case',
                     'Fitting_Time',
                     'URER',
                     'URETN',
                     'OutPath']
        self.configDict = dict.fromkeys(configKey)

    def setValue_Start_Time(self, value):
        self.configDict['Start_Time'] = value

    def setValue_Indicator(self, value):
        self.configDict['Indicator'] = value

    def setValue_Input_File(self, value):
        self.configDict['Input_File'] = value

    def setValue_Input_Type(self, value):
        self.configDict['Input_Type'] = value

    def setValue_Interval(self, value):
        self.configDict['Interval'] = value

    def setValue_Para_Num(self, value):
        self.configDict['Para_Num'] = value

    def setValue_Para_Case(self, value):
        self.configDict['Para_Case'] = value

    def setValue_Fitting_Time(self, value):
        self.configDict['Fitting_Time'] = value

    # def setValue_Orbit_Alt(self, value):
    #     self.configDict['Orbit_Alt'] = value

    def setValue_URER(self, value):
        self.configDict['URER'] = value

    def setValue_URETN(self, value):
        self.configDict['URETN'] = value

    def setValue_OutPath(self, value):
        self.configDict['OutPath'] = value

    def getValue_OutPath(self):
        return self.configDict['OutPath']

    def genConfigFile(self):
        with open('../config_nicegui.ini', 'w') as out:
            out.writelines('[config]\n')
            for key, value in self.configDict.items():
                oneLine = f"{key}    = {value}\n"
                out.writelines(oneLine)


class UREFile:
    def __init__(self):
        self.UREFilePath = ''
        self.RadioOption = ''

    def setValue_UREFile(self, value):
        self.UREFilePath = value

    def getValue_UREFile(self):
        return self.UREFilePath

    def setValue_RadioOption(self, value):
        self.RadioOption = value

    def getValue_RadioOption(self):
        return self.RadioOption


async def brdfit_main_gui(cfgPath):
    config.update('jax_enable_x64', True)
    opt = opttype()

    # (1) read config and sp3 file
    readConfig(cfgPath, opt)

    myTimePool, mySp3Pool = readsp3File(opt.inputfile, opt.indicator, opt.inputtype, opt.inter_proc)

    # (2) eph fit
    # toeSow, x_final_brd, RTN_lst = ephfit(myTimePool, mySp3Pool, opt)
    toeSow, x_final_brd, RTN_lst = await run.cpu_bound(ephfit, myTimePool, mySp3Pool, opt)

    # (3) output the result
    outputResult(toeSow, x_final_brd, RTN_lst, opt)


async def mycalonClick():
    n = ui.notification(message="Computing...", spinner=True, type="ongoing", timeout=None)
    await brdfit_main_gui('../config_nicegui.ini')
    # brdfit_main_gui('../config_nicegui.ini')
    n.message = "Done"
    n.spinner = False
    n.icon = 'done'
    n.timeout = 5.0


def plotURE(UREFile):
    data = np.loadtxt(UREFile, dtype=np.float64)
    x = data[:, 1] - data[0, 1]
    y = data[:, 2:]

    colors = ['red', 'green', 'blue', 'purple']  # choose color for each column
    markers = ['o', 's', '^', 'v']  # choose different markers for each column

    # plt.cla()
    # plt.clf()
    with ui.pyplot(figsize=(6, 4)):
        for i in range(len(colors)):
            plt.scatter(x, y[:, i], color=colors[i], marker=markers[i])

        plt.title('LEO satellite')

        legend_labels = ["R", "T", "N", "URE"]
        plt.legend(legend_labels, loc='best')

        plt.xlabel('Fitting Interval(s)')
        plt.ylabel('Fitting Error(m)')

        # plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
        # plt.show()


def find_free_port():
    min_port = 10000
    max_port = 65535
    for port in range(min_port, max_port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
            sock.close()
            return port
        except socket.error as e:
            continue
    raise Exception('No free ports found')


myConfig = GenConfig()
myUREFile = UREFile()

with ui.tabs() as tabs:
    tab1 = ui.tab('a', label='1. LEO satellite Option')
    tab2 = ui.tab('b', label='2. Fitting Option')
    tab3 = ui.tab('c', label='3. Output Option')
    tab4 = ui.tab('d', label='4. URE Error Visualization')

# (1) LEO satellite option
with ui.tab_panels(tabs, value='a').classes('w-full'):
    with ui.tab_panel('a'):
        with ui.card():
            # ui.button('1. LEO satellite Option')
            # ui.avatar('home')
            with ui.row():
                ui.input(label='Input_File', placeholder='',
                         on_change=lambda e: myConfig.setValue_Input_File(e.value),
                         validation={'Input too long': lambda value:
                         len(value) < 5000}).tooltip('example:/home/orbdata/aaa.sp3')
            with ui.row():
                names = ['P', 'PV']
                mySelectinputType = ui.select(options=names, with_input=True,
                                              on_change=lambda e: myConfig.setValue_Input_Type(e.value),
                                              label='Input_Type').classes('w-40')
            with ui.row():
                ui.input(label='Interval(s)', placeholder='',
                         on_change=lambda e: myConfig.setValue_Interval(e.value),
                         validation={'Input too long': lambda value:
                         len(value) < 5000})

            with ui.row():
                result = ui.input(label='Indicator', placeholder='',
                                  on_change=lambda e: myConfig.setValue_Indicator(e.value),
                                  validation={'Input too long': lambda value:
                                  len(value) < 5000})

            # with ui.row():
            #     ui.input(label='Orbit_Altitude(km)', placeholder='',
            #              on_change=lambda e: myConfig.setValue_Orbit_Alt(e.value),
            #              validation={'Input too long': lambda value:
            #              len(value) < 5000})

# (2) Fitting option
with ui.tab_panels(tabs, value='b').classes('w-full'):
    with ui.tab_panel('b'):
        with ui.card():
            # ui.avatar('home')
            with ui.row():
                ui.input(label='Start_Time', placeholder='2023 05 20 00 00 00.000',
                         on_change=lambda e: myConfig.setValue_Start_Time(e.value),
                         validation={'Input too long': lambda value:
                         len(value) < 5000}).tooltip('YYYY MM DD HH MM SS')

            with ui.row():
                ui.input(label='Fitting_Interval(s)', placeholder='',
                         on_change=lambda e: myConfig.setValue_Fitting_Time(e.value),
                         validation={'Input too long': lambda value:
                         len(value) < 5000})
            with ui.row():
                names = ['16', '17', '18', '19', '20', '21', '22']
                mySelect2 = ui.select(options=names, with_input=True, label='Para_Num',
                                      on_change=lambda e: myConfig.setValue_Para_Num(e.value)
                                      ).classes('w-40').tooltip('The number of parameters')
            with ui.row():
                names = ['case1', 'case2', 'case3']
                mySelect3 = ui.select(options=names, with_input=True, label='Para_Case',
                                      on_change=lambda e: myConfig.setValue_Para_Case(e.value)
                                      ).classes('w-40')

# (3) output option
with ui.tab_panels(tabs, value='c').classes('w-full'):
    with ui.tab_panel('c'):
        with ui.card():
            with ui.row():
                ui.input(label='URER', placeholder='',
                         on_change=lambda e: myConfig.setValue_URER(e.value),
                         validation={'Input too long': lambda value:
                         len(value) < 5000})

            with ui.row():
                ui.input(label='URETN', placeholder='',
                         on_change=lambda e: myConfig.setValue_URETN(e.value),
                         validation={'Input too long': lambda value:
                         len(value) < 5000})

            with ui.row():
                ui.input(label='OutPath', placeholder='',
                         on_change=lambda e: myConfig.setValue_OutPath(e.value),
                         validation={'Input too long': lambda value:
                         len(value) < 5000}).tooltip('example:/home/zhaox/output')

        with ui.row():
            def myokonClick():
                myConfig.genConfigFile()
                ui.notify('The config file has been generated!')


            ui.button('(a) OK', on_click=myokonClick)
            ui.button('(b) Calculate', on_click=mycalonClick)

# (4) Output URE
with ui.tab_panels(tabs, value='d').classes('w-full'):
    with ui.tab_panel('d'):
        with ui.card():
            def on_radio_change():
                if radio.value == 'New File':
                    outpath = myConfig.getValue_OutPath()
                    if len(outpath) == 0:
                        ui.notify('Please calculate before plotting!')
                        return
                    else:
                        myUREFile.setValue_UREFile(os.path.join(outpath, 'URE_file'))
                        ui.notify(myUREFile.UREFilePath)


            radio = ui.radio(['Old File', 'New File'], value='None',
                             on_change=lambda e: myUREFile.setValue_RadioOption(e.value)).classes('items-stretch')

            input_text = ui.input(label='Old URE File', placeholder='',
                                  on_change=lambda e: myUREFile.setValue_UREFile(e.value),
                                  validation={'Input too long': lambda value:
                                  len(value) < 5000}).tooltip('example:/home/zhaox/output/URE_file')


        def myplotClick():
            myradioOption = myUREFile.getValue_RadioOption()
            if myradioOption == 'New File':
                outpath = myConfig.getValue_OutPath()
                if outpath is None:
                    ui.notify('Please calculate before plotting!')
                    return
                else:
                    myUREFile.setValue_UREFile(os.path.join(outpath, 'URE_file'))
                    ui.notify(myUREFile.UREFilePath)

            outpath = myUREFile.getValue_UREFile()
            if len(outpath) == 0:
                ui.notify('Please provide URE File!')
                return
            plotURE(outpath)


        ui.button('PlotURE', on_click=myplotClick)

# ui.upload(on_upload=lambda e: ui.notify(f'Uploaded {e.name}')).classes('max-w-full')
#
# ui.date(value='2023-01-01', on_change=lambda e: result.set_text(e.value))
# result = ui.label()

# ui.run(title='LEOEPHFIT', port=10033)
ui.run(title='LEOEPHFIT', port=find_free_port())
