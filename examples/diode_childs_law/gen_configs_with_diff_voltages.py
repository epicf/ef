import os

basic_conf_file = 'diode_childs_law.conf'
with open( basic_conf_file, 'r') as f:
    basic_conf = f.readlines()

# todo: find instead of hardcoding
anode_voltage_linenum = 42
output_file_name_linenum = 66

voltages = list( range(0, 310, 30) )
volts_to_cgs = 1.0 / 300.0

for V in voltages:
    conf_name = "V" + "{0:0=3d}".format(V) + ".conf"
    anode_voltage_line = "inner_region_box_potential = +" + \
                         "{0:.5f}".format(V * volts_to_cgs ) + "\n"
    output_filename_prefix_line = "output_filename_prefix = V" + \
                                  "{0:0=3d}".format(V) + "_" + "\n"
    basic_conf[anode_voltage_linenum] = anode_voltage_line
    basic_conf[output_file_name_linenum] = output_filename_prefix_line
    with open( conf_name, 'w') as f:
        f.writelines( basic_conf )

    
