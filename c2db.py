from ase.db import connect
from operator import itemgetter
from collections import Counter

from ase.spacegroup import get_spacegroup

import argparse
import itertools
import numpy as np
import os

trans_metals = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg' ]

list_of_properties = [
	'A', 'E_B', 'E_x', 'E_y', 'E_z', 'J', 'N_nn', 'Topology', 'age', 'alphax', 'alphax_el',
    'alphax_lat', 'alphay', 'alphay_el', 'alphay_lat', 'alphaz', 'alphaz_el', 'alphaz_lat',
    'asr_id', 'c_11', 'c_12', 'c_13', 'c_21', 'c_22', 'c_23', 'c_31', 'c_32', 'c_33',
    'calculator', 'cbm', 'cbm_gw', 'cbm_hse', 'cell_area', 'charge', 'class', 'cod_id',
    'crystal_type', 'dE_zx', 'dE_zy', 'dim_nclusters_0D', 'dim_nclusters_1D', 'dim_nclusters_2D',
    'dim_nclusters_3D', 'dim_primary', 'dim_primary_score', 'dim_score_0123D', 'dim_score_012D',
    'dim_score_013D', 'dim_score_01D', 'dim_score_023D', 'dim_score_02D', 'dim_score_03D',
    'dim_score_0D', 'dim_score_123D', 'dim_score_12D', 'dim_score_13D', 'dim_score_1D',
    'dim_score_23D', 'dim_score_2D', 'dim_score_3D', 'dim_threshold_0D', 'dim_threshold_1D',
    'dim_threshold_2D', 'dipz', 'doi', 'dos_at_ef_nosoc', 'dos_at_ef_soc',
    'dynamic_stability_phonons', 'dynamic_stability_stiffness', 'efermi', 'ehull', 'emass_cb_dir1',
    'emass_cb_dir2', 'emass_vb_dir1', 'emass_vb_dir2', 'energy', 'etot', 'evac', 'evacdiff',
    'first_class_material', 'fmax', 'folder', 'formula', 'gap', 'gap_dir', 'gap_dir_gw',
    'gap_dir_hse', 'gap_dir_nosoc', 'gap_gw', 'gap_hse', 'gap_nosoc', 'has_inversion_symmetry',
    'hform', 'icsd_id', 'id', 'is_magnetic', 'lam', 'magmom', 'magstate', 'mass', 'minhessianeig',
    'natoms', 'nspins', 'pbc', 'plasmafrequency_x', 'plasmafrequency_y', 'pointgroup', 'smax',
    'spacegroup', 'speed_of_sound_x', 'speed_of_sound_y', 'spgnum', 'spin', 'spin_axis',
    'stoichiometry', 'thermodynamic_stability_level', 'uid', 'unique_id', 'user', 'vbm', 'vbm_gw',
    'vbm_hse', 'volume', 'workfunction'
    ]


def get_arguments():
	parser = argparse.ArgumentParser(description=
                                     'C2DB parser\n\n'
                                     
                                     'This script is design to read and retrieve cell vectors and atomic cartesian '
                                     'coordinates from the compounds of the c2db database. There are 5 arguments that '
                                     'can be provided to the script in order to configure the desirable query and to '
                                     'also manage the outputs:\n\n'

				     '-- tm: It\'s a options for selected only compound with transition metals'
				     'The mode true selected only compounds with transition metal.'
				     'The option False select materials that correspond with the query'                                     
 
                                     '-- query: It\'s the query for which the database will be parsed. The query '
                                     'should be presented as a single string of text with different parse arguments '
                                     'being separated by a comma and inside quotation marks. For example: '
                                     '\'stoichiometry=AB2, 0.5<gap_hse<3.5\', will retrieve the compounds with '
                                     'stoichiometry equals to AB2 and with band gap HSE06 between 0.5 and 3.5 eV.\n\n'
                                   
                                     '-- database: The name of the database to be parsed. The default name is '
                                     '\'c2db.db\', but a new name can be provided to match the name of the database in '
                                     'the working folder.\n\n '
                                     
                                     '-- key: Property to be retrieved from the compounds parsed by the query argument. '
                                     'The list of possible keys to be used can be found in '
                                     '\'https://cmr.fysik.dtu.dk/c2db/c2db.html\'. If a particular compound does not '
                                     'have the desired property, a value of NaN will be assigned to it. The resulting '
                                     'values will be saved in a folder named after the property and in a text file '
                                     'also named after the property. The text file will have the chemical formula of '
                                     'the respective compound and its property value in a two column scheme:\n\n'
                                     'Fe2Te2 \tproperty value\n'
                                     'Ag2O2  \tproperty value\n '
                                     '...\n\n'
                                     
                                     '-- folder: Name of the folder to save the results\n\n'
                                     
                                     '-- mode: The configuration of the saved file with cell vector and cartesian'
                                     'coordinates information. If mode = True (default), the chemical symbols will be'
                                     'written in the same line as its cartesian coordinates:\n\n'
                                     'Ag    x    y    z\n'
                                     'Ag    x    y    z\n'
                                     'O     x    y    z\n'
                                     'O     x    y    z\n\n'
                                     'If mode = False, the chemical symbol, along with its quantity, will be written '
                                     'first, followed by the cartesian coordinates:\n\n'
                                     'Ag  O\n'
                                     '2   2\n'
                                     '\tx    y    z\n'
                                     '\tx    y    z\n'
                                     '\tx    y    z\n'
                                     '\tx    y    z\n'
                                     ,
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )

	parser.add_argument('--query', nargs='+', help='Queries to select the compounds in the database')
	parser.add_argument('--tm', type=str, default='True', help='Select materials with transition metals')
	parser.add_argument('--database', type=str, default='c2db.db', help='The name of the database')
	parser.add_argument('--key', type=str, default=None, help='Property to select among the compounds '
                                                              'returned by the query')
	parser.add_argument('--folder', type=str, default='query', help='Folder to save the archives with information '
                                                                    'on the material cell, cartesian coordinates and '
                                                                    'selected property')
	parser.add_argument('--mode', type=str, default='True', help='The formatting of the final files')
	return parser.parse_args()


def main():
	args = get_arguments()

	if args.query is None:
		raise ValueError('A query value should be provided to parse the database. For example: '
                         '\'stoichiometry=AB2, 0.5<gap_hse<3.5\'')

	# connecting to the database
	db = connect(args.database)
	# querying the compounds
	db_select = db.select(args.query[0])

	print(args.folder)
	print(args.key)

	for row in db_select:
		for trans_m in trans_metals: 
			if trans_m in row.formula:
				print(row.formula, trans_m)
				pass
			else:
				idr = row.id

	# creates the folder to save the results if it does not exists
	path = args.folder
	if not os.path.exists(path):
		os.mkdir(path)

	# creates a folder to save the coordinates files
	coordinates_folder = os.path.join(path, 'coordinates')
	if not os.path.exists(coordinates_folder):
		os.mkdir(coordinates_folder)

		# creates ase.Atom object
		atoms = []
		for row in db_select:
			atoms.append(row.toatoms())

		# retrieving cell and coordinates
		for atom in atoms:
			symbols = atom.get_chemical_symbols()  # chemical symbols
			positions = atom.get_positions()  # cartesian coordinates
			cell = atom.get_cell().todict()['array']  # cell

			# sorting chemical symbols
			symbols_array = np.array(symbols)
			symbols_array = symbols_array[:, np.newaxis]
			sym_pos = np.concatenate([symbols_array, positions], axis=1)
			sym_pos = sorted(sym_pos, key=itemgetter(0))

			i=0
			for el in str(get_spacegroup(atom)).split()[:]:
				if el=="setting":
					break
				i+=1

			#archive_name = atom.get_chemical_formula()
			archive_name = "{}_{}".format(atom.get_chemical_formula(),"".join(str(get_spacegroup(atom)).split()[1:i]).replace("/", "!"))            

			with open(os.path.join(coordinates_folder, archive_name + '.txt'), 'w') as file:	
				file.write(str(len(symbols)) + '\n\n')  # number of atoms
				
				for row in cell:
					file.write(f'\t{row[0]:<30} {row[1]:<30} {row[2]}\n')  # cell vectors

				if args.mode.lower() == 'true':
					file.write('\n\n')

					for row in sym_pos:
						file.write(f'{row[0]:<5} {row[1]:<25} {row[2]:<25} {row[3]}\n')  # cartesian coordinates

				elif args.mode.lower() == 'false':
					file.write('\n\n')

					symbols_dict = sorted(Counter(symbols).most_common())
					file.write('  '.join([x[0] for x in symbols_dict]))
					file.write('\n')
					file.write('   '.join([str(x[1]) for x in symbols_dict]))
					file.write('\n\n')
					for row in sym_pos:
						file.write(f'{row[1]:<25} {row[2]:<25} {row[3]}\n')  # cartesian coordinates

				else:
					raise ValueError('The "mode" argument should either be "True" or "False"')

	if args.key is None:
		print('\nNo property was selected')
	else:
		if args.key not in list_of_properties:
			print(f'\nProperty "{args.key}" not found. The value should be one of the keys in the table '
				'"key-values pairs" in https://cmr.fysik.dtu.dk/c2db/c2db.html')
			args.key = None
		
		else:
			# creates a folder to save the property file
			properties_folder = os.path.join(path, args.key)
            
			if not os.path.exists(properties_folder):
				os.mkdir(properties_folder)

				labels = []
				key_property = []

				db_select = db.select(args.query[0])
				for row in db_select:
					tmp = dict(Counter(row.symbols))
					tmp = list(itertools.chain(*tmp.items()))
					labels.append(''.join([str(x) for x in tmp]))

					try:
						property_value = eval('row.' + args.key)
					except:
						property_value = np.nan

					key_property.append(property_value)
                    

					with open(properties_folder + '/' + args.key + '.txt', 'w') as file:
						for atom, prop in zip(labels, key_property):
							file.write(f'{atom:<10} {prop}\n')

if __name__ == '__main__':
	main()
