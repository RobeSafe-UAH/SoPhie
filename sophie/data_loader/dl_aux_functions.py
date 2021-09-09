def safe_list(input_data, warning=False, debug=False):
	safe_data = copy.copy(input_data)
	return safe_data

def safe_path(input_path, warning=False, debug=False):
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data

def isstring(string_test):
	return isinstance(string_test, str)

def find_unique_common_from_lists(input_list1, input_list2, only_com=False, warning=False, debug=False):
	input_list1 = safe_list(input_list1, warning=warning, debug=debug)
	input_list2 = safe_list(input_list2, warning=warning, debug=debug)

	common_list = list(set(input_list1).intersection(input_list2))

	if only_com: return common_list

	# find index
	index_list1 = []
	for index in range(len(input_list1)):
		item = input_list1[index]
		if item in common_list:
			index_list1.append(index)

	index_list2 = []
	for index in range(len(input_list2)):
		item = input_list2[index]
		if item in common_list:
			index_list2.append(index)

	return common_list, index_list1, index_list2

def remove_list_from_list(input_list, list_toremove_src, warning=False, debug=False):
	list_remained = safe_list(input_list, warning=warning, debug=debug)
	list_toremove = safe_list(list_toremove_src, warning=warning, debug=debug)
	list_removed = []
	for item in list_toremove:
		try:
			list_remained.remove(item)
			list_removed.append(item)
		except ValueError:
			if warning: print('Warning!!!!!! Item to remove is not in the list. Remove operation is not done.')

	return list_remained, list_removed

def load_list_from_folder(folder_path, ext_filter=None, depth=1, recursive=False, sort=True, save_path=None, debug=True):
    folder_path = safe_path(folder_path)
    if isstring(ext_filter): ext_filter = [ext_filter]                               # convert to a list
    # zxc

    fulllist = list()
    if depth is None:        # find all files recursively
        recursive = True
        wildcard_prefix = '**'
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = os.path.join(wildcard_prefix, '*' + string2ext_filter(ext_tmp))
                wildcard = os.path.join(wildcard_prefix, '*' + ext_tmp)
                curlist = glob2.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                fulllist += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob2.glob(os.path.join(folder_path, wildcard))
            if sort: curlist = sorted(curlist)
            fulllist += curlist
    else:                    # find files based on depth and recursive flag
        wildcard_prefix = '*'
        for index in range(depth-1): wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = wildcard_prefix + string2ext_filter(ext_tmp)
                wildcard = wildcard_prefix + ext_tmp
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                fulllist += curlist
            # zxc
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            # print(curlist)
            if sort: curlist = sorted(curlist)
            fulllist += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path, ext_filter=ext_filter, depth=depth-1, recursive=True)
            fulllist += newlist

    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)

    return fulllist, num_elem

def fileparts(input_path, warning=False, debug=False):
	good_path = safe_path(input_path, debug=debug)
	if len(good_path) == 0: return ('', '', '')
	if good_path[-1] == '/':
		if len(good_path) > 1: return (good_path[:-1], '', '')	# ignore the final '/'
		else: return (good_path, '', '')	                          # ignore the final '/'
	
	directory = os.path.dirname(os.path.abspath(good_path))
	filename = os.path.splitext(os.path.basename(good_path))[0]
	ext = os.path.splitext(good_path)[1]
	return (directory, filename, ext)

def seq_collate(data): # id_frame
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, idframe_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    id_frame = torch.cat(idframe_list, dim=0).permute(2, 0, 1)

    out = [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask, seq_start_end, id_frame]

    return tuple(out)

def get_folder_name(video_path, seq_name):
    town = str(int(seq_name/1000))
    seq = str(int(seq_name%1000))
    split = video_path.split('/')[-2].split('_')[-1]
    hd = (split == "test") and town == "10"
    folder = "Town{}{}_seq{}".format(
        town.zfill(2),
        "HD" if hd else "",
        seq.zfill(4)
    )
    full_path = os.path.join(video_path, folder)
    return full_path

def load_images(video_path, frames, extension="png", new_shape=(600,600)):
    frames_list = []
    cont = 0
    for frame in frames:
        folder_name = get_folder_name(video_path[0], frame[0].item())
        cont += 1
        image_id = str(int(frame[1].item()))
        image_url = os.path.join(folder_name, "{}.{}".format(image_id.zfill(6), extension))
        #print("image_url: ", image_url)
        frame = cv2.imread(image_url)
        frame = cv2.resize(frame, new_shape)
        frames_list.append(np.expand_dims(frame, axis=0))
    frames_arr = np.concatenate(frames_list, axis=0)
    return frames_arr

