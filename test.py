import os.path


finetune_model = r'C:\Users\admin\Downloads\scellseg-gui-main\output\fine-tune\finetune_scellseg_BBBC010_elegans'

a = os.path.isdir(finetune_model);
print(a)