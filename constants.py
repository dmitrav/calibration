import pandas
data = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/data/filtered_data_with_mz.csv', index_col=0).iloc[:, :2]
mz_dict = {k: v for k, v in zip(data['name'], data['mz'].astype('float'))}

# ML
seed = 2022
split_ratio = 0.8

n_jobs = -1
save_to = '/Users/andreidm/ETH/projects/calibration/res/'

# DATA
dilutions = ['0001', '0002', '0004', '0008', '0016', '0032', '0064']
aas = ['Proline', 'Asparagine', 'Lysine', 'Phenylalanine', 'Tyrosine']
pps = ['Cytosine', 'Uracil', 'Thymine', 'Adenine']

normalized_pp_metabolites = [
    'Acetone', 'Acetatic acid', 'C3:0 (Propionic acid)', 'Hydrogen sulfite', 'Succinic aldehyde', 'Methylpyrazine',
    'Hydroxypyridine', 'Silicic acid', 'Furfural', 'Cyclohexenone', 'Phosphoric acid', 'Oxobutanoic acid',
    'Pentanoic acid', 'Aminobutanoic acid (ABA)', 'Hydroxybutanoic acid', 'Benzyl alcohol', 'Aminophenol', 'Catechol',
    'N-Acetylimidazole', '2,3,5-Trimethylfuran', 'Cytosine', 'Uracil', 'Creatinine', '2-Hydroxy-2,4-pentadienoic acid',
    'Dihydrouracil', 'Proline', 'Fumarate', 'Valine; Betaine', 'Succinate', 'Threonine', 'Phenylacetaldehyde', 'Taurine',
    '2-Aminoethylphosphonate', 'Thymine', 'Dihydro-4,4-dimethyl-2,3-furandione', 'Dihydrothymine', 'Oxoproline',
    '4-Methyl-2-oxopentanoate', 'Ethyl isovalerate', '(Iso)Leucine', 'Asparagine', 'Ornithine', 'Heptane-1-thiol',
    '3,3-Dimethyl-1,2-dithiolane', 'Adenine', '(3Z,6Z)-3,6-Nonadien-1-ol', 'Ethosuximide',
    '(2-hydroxyethoxy)sulfonic acid', 'Octanoic acid', '3-Methyleneoxindole', '4-Acetamidobutanoate', 'Glutamine',
    'Lysine', 'Glutamate', 'Methionine', 'Diisopropyl disulfide', 'Hydrocinnamic acid', 'Thymol',
    '1,2-Benzisothiazol-3(2H)-one', 'Hydroxytyrosol', 'Histidine', '8-Hydroxy-5,6-octadienoic acid', 'Isopropylmaleate',
    'cis-4-Hydroxycyclohexylacetic acid', 'Oenanthic ether', '2-Naphthalenethiol__Oxoadipate',
    'Propylhydroxypentanoic acid', 'Coumarate', 'Deoxyhexose', 'Phenylalanine',
    'Pentonic acid__S-Propyl 1-propanesulfinothioate', 'Methyl beta-naphthyl ketone', 'C10:1', '2-Octenedioic acid',
    '2-Ethoxynaphthalene__Glycylproline', 'Tetrahydrofurfuryl butyrate', 'Decanoic acid (FA10:0)', 'Phenol sulfate',
    'Suberic acid', 'Arginine', '3-Propylmalate', 'Ethyl cinnamate', 'Serotonin', '4-Methoxycinnamic acid',
    '5-Phenylvaleric acid', 'Hexose__Theophylline', '2-Methyl-3-hydroxy-5-formylpyridine-4-carboxylate__Acamprosate',
    'Tyrosine', '5-exo-Hydroxy-1,2-campholide', 'Citronellyl formate', '3-Oxodecanoic acid', 'Azelaic acid',
    '3-Dehydroquinate', '3-Hydroxysuberic acid', 'C12:4', 'C12:1',
    '3,5,6-Trihydroxy-5-(hydroxymethyl)-2-methoxy-2-cyclohexen-1-one', 'Tryptophan',
    '2-Ethyl-1-hexanol sulfate__Sinapyl alcohol', 'C13:2', '3,4-Methyleneazelaic acid',
    '3-Hydroxysebacic acid', 'Epoxyeremopetasinorol', 'C14:0', 'Hydralazine pyruvate hydrazone',
    'Ethyl vanillin isobutyrate', 'Geranyl acetoacetate', 'Genipinic acid', '2-Carboxy-4-dodecanolide',
    '2-Hydroxymyristic acid', '3-Hydroxydodecanedioic acid', 'Gemfibrozil',
    '2,2,4,4-Tetramethyl-6-(1-oxobutyl)-1,3,5-cyclohexanetrione', 'Hypogeic acid', 'Alpha-Linolenic acid', 'C18:2',
    'Hexadecanedioic acid', 'Benzoyl ecgonine', '13-OxoODE', '13S-hydroxyoctadecadienoic acid',
    'Nonadeca-10(Z)-enoic acid', 'Toxin T2 tetrol', '(R)-3-Hydroxy-Octadecanoic acid', 'Acetylglucosamine sulfate',
    'Eicosapentaenoic acid', 'Arachidonic acid', 'C20:1', 'N-Undecylbenzenesulfonic acid',
    '13-L-Hydroperoxylinoleic acid', '2-Dodecylbenzenesulfonic acid', 'C25:0', 'MG(0:0/20:1(11Z)/0:0)',
    'MG(22:1(13Z)/0:0/0:0)', 'Ethyl beta-D-glucopyranoside', 'Eremopetasinorol', 'Vanilpyruvic acid', 'Cucurbic acid',
    '6-[(3,3-dimethyloxiran-2-yl)methyl]-5-hydroxy-7-methoxy-2-(3,4,5-trimethoxyphenyl)-3,4-dihydro-2H-1-benzopyran-4-one',
    '5-Hexyltetrahydro-2-oxo-3-furancarboxylic acid', 'Annuionone B'
]

normalized_pp_outliers = [
    'Thiodiglycol', 'Dihydropteridine', 'Tetrahydropteridine', '4-Amino-5-hydroxymethyl-2-methylpyrimidine',
    'Gentisic acid', 'C9:2', 'Hydroxyadipate__Methomyl', 'Phthalate', '5-Nitrosalicylate', '(Iso)Citrate',
    'Hydralazine acetone hydrazone__cis-4-Decenedioic acid', 'Dodecanoic acid (FA12:0)', 'Sebacic acid',
    '4-(1,1,3,3-Tetramethylbutyl)-phenol', '3-Oxododecanoic acid', 'C13:0', 'Methyl dihydrojasmonate',
    'C14:1', 'Propyl propane thiosulfonate', 'Heptyl 4-hydroxybenzoate', 'Equol', 'C15:0', 'N-Undecanoylglycine',
    '1,11-Undecanedicarboxylic acid', '5-Nonyltetrahydro-2-oxo-3-furancarboxylic acid',
    '8-Oxohexadecanoic acid', '7(14)-Bisabolene-2,3,10,11-tetrol', '1-(4-Hydroxy-3-methoxyphenyl)-3-decanone',
    '3D,7D,11D-Phytanic acid', 'Hydrocortamate']  # correct trends with outliers

normalized_aa_outliers = [
    # crazy trends
    'Acetatic acid', 'Phosphoric acid', 'Cytosine', 'Thiodiglycol', 'Tetrahydropteridine',
    '4-Amino-5-hydroxymethyl-2-methylpyrimidine', 'C9:2', 'Oenanthic ether', 'Tetrahydrofurfuryl butyrate',
    'Decanoic acid (FA10:0)', 'Phenol sulfate', '3-Propylmalate', 'C12:4', 'Dodecanoic acid (FA12:0)', 'C13:0', 'C14:0',
    'Heptyl 4-hydroxybenzoate', 'C15:0', '5-Nonyltetrahydro-2-oxo-3-furancarboxylic acid', '8-Oxohexadecanoic acid',
    '7(14)-Bisabolene-2,3,10,11-tetrol', '1-(4-Hydroxy-3-methoxyphenyl)-3-decanone', '13S-hydroxyoctadecadienoic acid',
    'Eicosapentaenoic acid', 'C25:0', 'MG(0:0/20:1(11Z)/0:0)', 'MG(22:1(13Z)/0:0/0:0)', 'Hydrocortamate',
    # correct trends with outliers
    'Hydrogen sulfite', 'Succinic aldehyde', 'Furfural', 'Ethosuximide', '(2-hydroxyethoxy)sulfonic acid',
    'Gentisic acid', '2-Naphthalenethiol__Oxoadipate', 'Propylhydroxypentanoic acid', 'Phthalate', '(Iso)Citrate',
    '1,11-Undecanedicarboxylic acid'
]

normalized_aa_metabolites = [x for x in [*normalized_pp_metabolites, *normalized_pp_outliers] if x not in normalized_aa_outliers]

initial_pp_metabolites = [
    'Cytosine', 'Uracil', 'Thymine', 'Dihydro-4,4-dimethyl-2,3-furandione', '3,3-Dimethyl-1,2-dithiolane', 'Adenine',
    'Tetrahydropteridine', 'Hydrocinnamic acid', '1,2-Benzisothiazol-3(2H)-one', 'C13:2'
]
# crazy trends
initial_pp_outliers = [x for x in [*normalized_pp_metabolites, *normalized_pp_outliers] if x not in initial_pp_metabolites]

initial_aa_metabolites = [
    'Methylpyrazine', 'Aminobutanoic acid (ABA)', 'N-Acetylimidazole', 'Dihydrouracil', 'Proline', 'Fumarate',
    'Valine; Betaine', 'Threonine', 'Phenylacetaldehyde', 'Dihydro-4,4-dimethyl-2,3-furandione', 'Dihydrothymine',
    'Oxoproline', '(Iso)Leucine', 'Asparagine', 'Ornithine', 'Heptane-1-thiol', 'Glutamine', 'Lysine', 'Glutamate',
    'Methionine', 'Diisopropyl disulfide', 'Histidine', '8-Hydroxy-5,6-octadienoic acid',
    'cis-4-Hydroxycyclohexylacetic acid', 'Coumarate', 'Phenylalanine', 'Suberic acid', 'Arginine',
    '2-Methyl-3-hydroxy-5-formylpyridine-4-carboxylate__Acamprosate', 'Tyrosine',
    '3,5,6-Trihydroxy-5-(hydroxymethyl)-2-methoxy-2-cyclohexen-1-one', 'Tryptophan'
]
# crazy trends
initial_aa_outliers = [x for x in [*normalized_pp_metabolites, *normalized_pp_outliers] if x not in initial_aa_metabolites]
