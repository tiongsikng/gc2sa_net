# contains directories of database

main_path = { 'main' : '/home/tiongsik/Python/conditional_biometrics/data' }

evaluation = { 'identification' : main_path['main'] + '/', 'verification' : main_path['main'] }

trainingdb = { 'db_name' : 'trainingdb', 'face_train' : main_path['main'] + '/trainingdb/train/face', 'peri_train' : main_path['main'] + '/trainingdb/train/peri', 'face_val' : main_path['main'] + '/trainingdb/val/face', 'peri_val' : main_path['main'] + '/trainingdb/val/peri', 'face_test' : main_path['main'] + '/trainingdb/test/face', 'peri_test' : main_path['main'] + '/trainingdb/test/peri' }

ethnic = { 'db_name' : 'ethnic',  
'face_gallery' : main_path['main'] + '/ethnic/Recognition/gallery/face',  'peri_gallery' : main_path['main'] + '/ethnic/Recognition/gallery/peri',  
'face_probe' : main_path['main'] + '/ethnic/Recognition/probe/face',  'peri_probe' : main_path['main'] + '/ethnic/Recognition/probe/peri', 
'face_case1_gen' : main_path['main'] + '/ethnic/Verification/genuine/case1/face', 'face_case2_gen' : main_path['main'] + '/ethnic/Verification/genuine/case2/face',
'face_case1_imp' : main_path['main'] + '/ethnic/Verification/impostor/case1/face', 'face_case2_imp' : main_path['main'] + '/ethnic/Verification/impostor/case2/face',
'peri_case1_gen' : main_path['main'] + '/ethnic/Verification/genuine/case1/peri', 'peri_case2_gen' : main_path['main'] + '/ethnic/Verification/genuine/case2/peri',
'peri_case1_imp' : main_path['main'] + '/ethnic/Verification/impostor/case1/peri', 'peri_case2_imp' : main_path['main'] + '/ethnic/Verification/impostor/case2/peri' }

pubfig = { 'db_name' : 'pubfig',
'face_gallery' : main_path['main'] + '/pubfig/gallery/face', 'peri_gallery' : main_path['main'] + '/pubfig/gallery/peri',
'face_probe1' : main_path['main'] + '/pubfig/probe1/face', 'peri_probe1' : main_path['main'] + '/pubfig/probe1/peri',
'face_probe2' : main_path['main'] + '/pubfig/probe2/face', 'peri_probe2' : main_path['main'] + '/pubfig/probe2/peri',
'face_probe3' : main_path['main'] + '/pubfig/probe3/face', 'peri_gallery' : main_path['main'] + '/pubfig/probe3/peri' }

facescrub = { 'db_name' : 'facescrub',
'face_gallery' : main_path['main'] + '/facescrub/gallery/face', 'peri_gallery' : main_path['main'] + '/facescrub/gallery/peri',
'face_probe1' : main_path['main'] + '/facescrub/probe1/face', 'peri_probe1' : main_path['main'] + '/facescrub/probe1/peri',
'face_probe2' : main_path['main'] + '/facescrub/probe2/face', 'peri_probe2' : main_path['main'] + '/facescrub/probe2/peri' }

imdb_wiki = { 'db_name' : 'imdb_wiki',
'face_gallery' : main_path['main'] + '/imdb_wiki/gallery/face', 'peri_gallery' : main_path['main'] + '/imdb_wiki/gallery/peri',
'face_probe1' : main_path['main'] + '/imdb_wiki/probe1/face', 'peri_probe1' : main_path['main'] + '/imdb_wiki/probe1/peri',
'face_probe2' : main_path['main'] + '/imdb_wiki/probe2/face', 'peri_probe2' : main_path['main'] + '/imdb_wiki/probe2/peri',
'face_probe3' : main_path['main'] + '/imdb_wiki/probe3/face', 'peri_gallery' : main_path['main'] + '/imdb_wiki/probe3/peri' }

ar = { 'db_name' : 'ar',
'face_gallery' : main_path['main'] + '/ar/gallery/face', 'peri_gallery' : main_path['main'] + '/ar/gallery/peri',
'face_blur' : main_path['main'] + '/ar/blur/face', 'peri_blur' : main_path['main'] + '/ar/blur/peri',
'face_exp_illum' : main_path['main'] + '/ar/exp_illum/face', 'peri_exp_illum' : main_path['main'] + '/ar/exp_illum/peri',
'face_occlude' : main_path['main'] + '/ar/occlude/face', 'peri_occlude' : main_path['main'] + '/ar/occlude/peri',
'face_scarf' : main_path['main'] + '/ar/scarf/face', 'peri_scarf' : main_path['main'] + '/ar/scarf/peri'
}