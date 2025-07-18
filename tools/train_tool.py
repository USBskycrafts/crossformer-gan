import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import shutil
from timeit import default_timer as timer

from tools.eval_tool import valid, gen_time_str, output_value
from tools.init_tool import init_test_dataset, init_formatter

logger = logging.getLogger(__name__)


def checkpoint(filename, playground, trained_epoch, config, global_step):
    models = playground.models
    optimizers = playground.optimizers
    save_params = {
        **{name: model.state_dict() for name, model in models.items()},
        **{optimizer.__class__.__name__ + name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }

    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning(
            "Cannot save models with error %s, continue anyway" % str(e))


def train(parameters, config, gpu_list, do_test=False):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = os.path.join(config.get(
        "output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        logger.warning(
            "Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"] + 1
    playground = parameters["playground"]
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    if do_test:
        init_formatter(config, ["test"])
        test_dataset = init_test_dataset(config)

    if trained_epoch == 0:
        shutil.rmtree(
            os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")), True)

    os.makedirs(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                exist_ok=True)

    logger.info("Training start....")

    output_value("Epoch",  "Stage", "Iterations", "Time Usage",
                 "Loss",   "Output Information", '\n', config)

    total_len = len(dataset)
    more = ""
    if total_len < 10000:
        more = "\t"
    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num

        acc_result = None
        total_loss = 0

        output_info = ""
        step = -1
        playground.train_step = 0
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])
            results = playground._train(
                data, config, gpu_list, acc_result, "train")

            loss, acc_result = results["loss"], results["acc_result"]
            total_loss += float(loss)

            if step % output_time == 0:
                output_info = output_function(acc_result, config)

                delta_t = timer() - start_time

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

            global_step += 1

        output_info = output_function(acc_result, config)
        delta_t = timer() - start_time
        output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
            "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

        if step == -1:
            logger.error(
                "There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        # TODO: warning: define should be modified
        checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), playground, current_epoch, config,
                   global_step)

        if current_epoch % test_time == 0:
            with torch.no_grad():
                # TODO: warning: define should be modified
                valid(playground, parameters["valid_dataset"],
                      current_epoch, config, gpu_list, output_function)
                if do_test:
                    # TODO: warning: define should be modified
                    valid(playground, test_dataset, current_epoch,
                          config, gpu_list, output_function, mode="test")
