def setFileDirectory(root_directory, foldername):
	import os, shutil
	folder_path = root_directory + '/' + str(foldername)#+ title.replace(' ', '_')
	try:
		if not os.path.isdir(root_directory):
			print('Error: saving directory does not exist.')
			exit()
		if os.path.exists(folder_path):
			shutil.rmtree(folder_path)
		os.makedirs(folder_path)
		print('Directory \"'+folder_path+'\" made.')
	except Exception as e:
		print('Error in making the movie folder.')
		print(e)
		exit()

	return folder_path