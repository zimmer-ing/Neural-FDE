import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

def plot_predictions(model, dataloader, device, num_samples=1,title='no title',path=None,show=True,raw_predictions=False,homogeneous=False):
    model.eval()
    model.to(device)

    y_hat,y,x,ts,z=model.predict(dataloader,samples_only=True,return_raw=raw_predictions)
    if not raw_predictions:
        # show the first sample
        time = ts.cpu().numpy().squeeze()
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        y_hat = y_hat.cpu().numpy()
    else:
        #create empty arrays
        time_np = []
        x_np = []
        y_np = []
        y_hat_np = []
        #iterate over the samples
        for i in range(len(ts)):
            time_np.append(ts[i].cpu().numpy().squeeze())
            x_np.append(x[i].cpu().numpy())
            y_np.append(y[i].cpu().numpy())
            y_hat_np.append(y_hat[i].cpu().numpy())
        time=time_np
        x=x_np
        y=y_np
        y_hat=y_hat_np



    # cut off samples otherwise plot is too big
    time = time[0:num_samples]
    x = x[0:num_samples]
    y = y[0:num_samples]
    y_hat = y_hat[0:num_samples]



    # plot the results
    fig = go.Figure()
    fig.update_layout(title_text=title)

    # Check if x, y, y_hat are lists
    is_list_x = isinstance(x, list)
    is_list_y = isinstance(y, list)
    is_list_y_hat = isinstance(y_hat, list)

    for i in range(num_samples):

        if not homogeneous:
            # Plot inputs
            num_x = x[0].shape[-1] if is_list_x else x.shape[-1]
            for j in range(num_x):
                x_data = x[i][:,:, j] if is_list_x else x[i, :, j]
                fig.add_trace(go.Scatter(x=time[i], y=x_data.flatten(), mode='lines', name=f'Sample_{i}_x{j}'))

        # Plot outputs
        num_y = y[0].shape[-1] if is_list_y else y.shape[-1]
        for j in range(num_y):
            y_data = y[i][:,:,j] if is_list_y else y[i, :, j]
            fig.add_trace(go.Scatter(x=time[i], y=y_data.flatten(), mode='lines', name=f'Sample_{i}_y{j}'))

        # Plot predictions
        num_y_hat = y_hat[0].shape[-1] if is_list_y_hat else y_hat.shape[-1]
        for j in range(num_y_hat):
            y_hat_data = y_hat[i][:,:, j] if is_list_y_hat else y_hat[i, :, j]
            fig.add_trace(go.Scatter(x=time[i], y=y_hat_data.flatten(), mode='lines', name=f'Sample_{i}_y_hat{j}'))

    if show:
        fig.show()

    if path is not None:
        #check if path exists
        if not path.exists():
            path.mkdir(parents=True)
        #save fig
        fig.write_html(str(path / (title + '.html')))








def dl_is_shuffle(dataloader):
    # Check the type of sampler being used by the DataLoader
    if isinstance(dataloader.sampler, RandomSampler):
        return True
    elif isinstance(dataloader.sampler, SequentialSampler):
        return False
    # For BatchSampler, check if the underlying sampler is a RandomSampler
    elif hasattr(dataloader, 'batch_sampler') and isinstance(dataloader.batch_sampler.sampler, RandomSampler):
        return True
    else:
        # If none of the above, it's unclear or not shuffled
        return False


def change_to_sequential_and_return_new(dataloader):
    # Create a new DataLoader with SequentialSampler
    new_dataloader = DataLoader(
        dataset=dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,  # Set shuffle to False to use SequentialSampler
        sampler=SequentialSampler(dataloader.dataset),  # Explicitly set to SequentialSampler
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers
    )

    # Check if the new DataLoader is in sequential mode using the is_shuffle function
    if not dl_is_shuffle(new_dataloader):
        print("Successfully created a new DataLoader in sequential mode.")
    else:
        print("Failed to create a new DataLoader in sequential mode.")

    return new_dataloader

def ensure_sequential_dataloader(dataloader):
    # Check if the DataLoader is already in sequential mode
    if not dl_is_shuffle(dataloader):
        print("DataLoader is already in sequential mode.")
        return dataloader

    # Create a new DataLoader with SequentialSampler
    new_dataloader = change_to_sequential_and_return_new(dataloader)

    return new_dataloader


import torch


def concatenate_batches(predictions):
    """
    Concatenate a list of batched tensors into a single tensor with shape [time, features].

    Parameters:
    - predictions: A list of tensors with shape [batch, time, features].

    Returns:
    - A tensor with shape [total_time, features], where total_time is the sum of all time dimensions
      across the batches.
    """
    # Step 1: Concatenate all tensors in the list along the batch dimension
    concatenated = torch.cat(predictions, dim=0)  # Results in [total_batches * batch, time, features]

    if len(concatenated.shape) == 2:
        return concatenated
    # Calculate the new shape
    total_batches_times_batch, time, features = concatenated.shape

    # Step 2: Reshape to [total_time, features], merging the first two dimensions
    result = concatenated.view(-1, features)  # Merges batch and time dimensions

    return result




if __name__ == "__main__":

    #test the functions above
    # Example usage
    dataset = torch.rand(100, 2)  # Just a dummy dataset
    dataloader_shuffled = DataLoader(dataset, batch_size=10, shuffle=True)
    dataloader_sequential = DataLoader(dataset, batch_size=10, shuffle=False)

    print("Shuffled DataLoader:", dl_is_shuffle(dataloader_shuffled))
    print("Sequential DataLoader:", dl_is_shuffle(dataloader_sequential))

    # Example usage
    dataset = torch.rand(100, 2)  # Dummy dataset
    dataloader_shuffled = DataLoader(dataset, batch_size=10, shuffle=True)

    # Change DataLoader to sequential and check
    print("DataLoader is shuffle before :", dl_is_shuffle(dataloader_shuffled))
    dataloader_sequential = ensure_sequential_dataloader(dataloader_shuffled)
    print("DataLoader is shuffle after :", dl_is_shuffle(dataloader_sequential))

    ##### concat_batches function
    # Example usage
    # Single batch case, batch dimension size = 1
    batch1 = torch.randn(1, 5, 3)  # For example, 1 sequence of length 5 with 3 features
    # Multiple batch case
    batch2 = torch.randn(2, 5, 3)  # For example, 2 sequences of length 5 with 3 features each

    # Concatenate batches
    result_single = concatenate_batches([batch1])  # Handling single list element
    result_multiple = concatenate_batches([batch1, batch2])  # Handling multiple batches

    print("Single batch result shape:", result_single.shape)
    print("Multiple batches result shape:", result_multiple.shape)