from flops_counter import get_model_complexity_info
import models as models


net = models.__dict__['PreActResNet101'](num_classes=100)
net = models.__dict__['mylayer'](group=64)

flops, params = get_model_complexity_info(net, (64, 32, 32), as_strings=True, print_per_layer_stat=False)
print('Flops: ' + flops)
print('params: ' + params)
