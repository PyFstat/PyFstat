basic_sft = data/H-4800_H1_1800SFT_basic-1000000000-8640000.sft
glitch_sft = data/H-4800_H1_1800SFT_glitch-1000000000-8640000.sft

data/fully_coherent_corner.png : $(basic_sft) fully_coherent_search.py
	python fully_coherent_search.py

$(basic_sft) $(glitch_sft): make_fake_data.py
	python make_fake_data.py
