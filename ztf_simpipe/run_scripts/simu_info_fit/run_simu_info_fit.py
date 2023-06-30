import os
from ztf_pipeutils.ztf_util import make_dict_from_config, make_dict_from_optparse
from optparse import OptionParser, OptionGroup
import ztf_simfit_input as simfit_input


def cmd(script, confDict):
    for key, vals in confDict.items():
        vv = eval('opts.{}'.format(key))
        script += ' --{} {}'.format(key, vv)

    return script


def fill_grp(group, confDict):

    # parser : 'dynamical' generation
    for key, vals in confDict.items():
        vv = vals[1]
        if vals[0] != 'str':
            vv = eval('{}({})'.format(vals[0], vals[1]))
        group.add_option('--{}'.format(key), help='{} [%default]'.format(
            vals[2]), default=vv, type=vals[0], metavar='')


parser = OptionParser()

# parser for simulation
path = list(simfit_input.__path__)
confDict_simu = make_dict_from_config(path[0], 'config_simu.txt')
group_simu = OptionGroup(parser, "Simulation")
parser.add_option_group(group_simu)
fill_grp(group_simu, confDict_simu)

# parser for info
confDict_info = make_dict_from_config(path[0], 'config_info.txt')
group_info = OptionGroup(parser, "Info")
parser.add_option_group(group_info)
fill_grp(group_info, confDict_info)

# parser for fit
confDict_fit = make_dict_from_config(path[0], 'config_fit_lc.txt')
group_fit = OptionGroup(parser, "Fit LC")
parser.add_option_group(group_fit)
fill_grp(group_fit, confDict_fit)

opts, args = parser.parse_args()


# Now run!
# simulation
print('Simulation in progress...')
simu_cmd = cmd(
    'python run_scripts/simulation/run_simulation.py', confDict_simu)
os.system(simu_cmd)

# info
print('Info in progress...')
info_cmd = cmd('python run_scripts/info/run_info.py', confDict_info)
os.system(info_cmd)

# fit lc
print('Fit LC in progress...')
fit_cmd = cmd('python run_scripts/fit_lc/run_fit_lc.py', confDict_fit)
os.system(fit_cmd)


"""
print('hhh', script)

print(test)

ntransient = 88
nproc = 8

# simulation
simu_script = 'ztf_stage/script/run_simulation.py'
simu_cmd = 'python {}  --ntransient {} --nproc {}'.format(
    simu_script, ntransient, nproc)
os.system(simu_cmd)

# info+selec
info_script = 'ztf_stage/script/run_info.py'
info_cmd = 'python {}'.format(info_script)
os.system(info_cmd)

# fit
fit_script = 'ztf_stage/script/run_fit_lc.py'
fit_cmd = 'python {} --nproc {}'.format(fit_script, ntransient, nproc)
os.system(fit_cmd)
"""
