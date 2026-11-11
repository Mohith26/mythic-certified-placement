import pickle, struct, json, collections, io
f=open('Epoch99.pth','rb')
class Stor:
    def __init__(s,key,dtype,loc,size): s.key=key; s.dtype=dtype; s.size=size
storages={}
class FakeType:
    def __init__(s,name): s.name=name
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod=='collections' and name=='OrderedDict': return collections.OrderedDict
        if mod=='torch._utils' and name=='_rebuild_tensor_v2':
            def rb(storage, offset, size, stride, requires_grad=False, backward_hooks=None, *a):
                return ('T', storage, offset, tuple(size), tuple(stride))
            return rb
        if mod=='torch' and name.endswith('Storage'): return FakeType(name)
        raise Exception('unknown '+mod+'.'+name)
    def persistent_load(self, pid):
        typ, stype, key, loc, size = pid[0], pid[1], pid[2], pid[3], pid[4]
        if key not in storages: storages[key]=Stor(key,stype.name,loc,size)
        return storages[key]
magic=pickle.load(f); proto=pickle.load(f); sysinfo=pickle.load(f)
print(proto, sysinfo)
sd=U(f).load()
keys=pickle.load(f)
print(keys)
data={}
for k in keys:
    n=struct.unpack('<q', f.read(8))[0]
    st=storages[k]
    assert st.dtype=='FloatStorage', st.dtype
    data[k]=struct.unpack('<%df'%n, f.read(4*n))
out={}
for name,(T,stor,off,size,stride) in sd.items():
    n=1
    for s in size: n*=s
    vals=data[stor.key][off:off+n]
    out[name]={'shape':list(size),'data':[round(v,7) for v in vals]}
    print(name,size,stride)
json.dump(out,open('zerodce_weights.json','w'))
