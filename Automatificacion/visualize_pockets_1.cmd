open 2lzm.pdb
surface #0
surfrepr mesh #0

open predicted_pockets_1.pdb
select #1 & :POC
rep sphere sel
color yellow sel
~select
focus

open pocket_1_residues_1.pdb
color red #2
surface #2

open pocket_2_residues_1.pdb
color green #3
surface #3

open pocket_3_residues_1.pdb
color blue #4
surface #4

open pocket_4_residues_1.pdb
color magenta #5
surface #5

open pocket_5_residues_1.pdb
color cyan #6
surface #6

open pocket_6_residues_1.pdb
color orange #7
surface #7

focus
