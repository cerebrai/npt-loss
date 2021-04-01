import os
import torch
import struct


def write_bin(name, prefix, path, features):
    for i, feature in enumerate(features):

        _path = path[i].split('/')
        #if len(_path)==3:
        a=os.path.join(_path[-3], _path[-2])
        b=_path[-1]
        #else:
        #   a,b = _path[-2], _path[-1]
        out_dir = os.path.join(prefix, a)
        if not os.path.exists(out_dir):
           os.makedirs(out_dir)
        out_file = os.path.join(out_dir, b+"_%s.bin"%(name))
#        print('out_file:{}'.format(out_file))
        feature = list(feature)
        with open(out_file, 'wb') as f:
           f.write(struct.pack('4i', len(feature),1,4,5))
           f.write(struct.pack("%df"%len(feature), *feature))

def mega_extract(name, prefix, test_loader, model):

    with torch.no_grad():
        for i, (input, flst) in enumerate(test_loader):
#            print('input:{}'.format(type(input)))
            output = model(input.cuda())+model(torch.flip(input.cuda(),[3]))
            norm = output.pow(2).sum(dim=1, keepdim=True).sqrt()+1e-10
            output = torch.div(output,norm)
            #pudb.set_trace()

            write_bin(name, prefix, flst, output)

#            print("=" * 60)
            print('Batch {}/{}'.format(i + 1, len(test_loader)))
#            print("=" * 60)




