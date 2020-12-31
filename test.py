from obj_loader import Obj_Loader

obj_loader = Obj_Loader(aligned = True)
print(len(obj_loader))
print(type(obj_loader[0]))
for i in range(len(obj_loader)):
	print(obj_loader[i].shape)