from tqdm.auto import tqdm
import random
import torch
import csv

# Train model
def train(model, criterion, epochs, opt, train_dl, val_dl, noise_levels, score=False, device="cuda"):
   
    # Lists to track training progress
    train_losses = []
    validation_losses = []
    train_accs = []
    validation_accs = []

    cols = ["train_loss", "val_loss", "train_acc", "val_acc"]
    data_path = 'data/model_chk/results.csv'
    with open(data_path,'w') as f:
        writer = csv.writer(f)
        writer.writerow(cols)

    
    print('Training progress:')
    for epoch in tqdm(range(epochs)):   # Show progress bar with tqdm
        
        if epoch % 10 ==0:
            print('Epoch {}/{}'.format(epoch+1, epochs))
        model.train()
        total_correct = 0
        train_loss = 0
        samples = 0

        # Iterate through batches, train model
        for x_train, y_train in train_dl:
            n = x_train.shape[0]
            # choose random noise level
            lvl_idx = torch.randint(0, noise_levels, (n,), device=device)
            model_input = torch.stack([x_train[i,lvl_idx[i],...] for i in range(n)])
            if score:
                lvl_input = lvl_idx / noise_levels
                pred = model(model_input, lvl_input)
            else:
                pred = model(model_input)
            
            # Give higher weight to later noise levels
            loss = torch.mean(criterion(pred, y_train) * (lvl_idx + 1))

            loss.backward()
            opt.step()
            opt.zero_grad()

            # Calculate batch accuracy
            confidence, predicted = torch.max(pred.data, 1)
            correct = (predicted == y_train).sum().item()
            
            train_loss += loss.item()
            total_correct += correct
            samples += n

        train_acc = total_correct/samples
        train_loss = train_loss/len(train_dl)

        # Use validation data to check for overfitting
        model.eval()
        with torch.no_grad():
            val_loss = 0
            total_correct = 0
            samples = 0
            for x_val, y_val in val_dl:
                n = x_val.shape[0]
                # evaluate at all noise levels
                for noise_level in range(noise_levels):
                    model_input = x_val[:,noise_level,...]
                    if score:
                        lvl_input = torch.ones(n, device=device) / noise_levels
                        pred = model(model_input, lvl_input)
                    else:
                        pred = model(model_input)
                        
                    loss = torch.mean(criterion(pred, y_val) * (noise_level+1))

                    confidence, predicted = torch.max(pred.data, 1)
                    correct = (predicted == y_val).sum().item()

                    val_loss += loss.item()
                    total_correct += correct
                    samples += n

        val_acc = total_correct / samples
        val_loss = val_loss / (len(val_dl)*noise_levels)
        
        train_losses.append(train_loss)
        validation_losses.append(val_loss)
        train_accs.append(train_acc)
        validation_accs.append(val_acc)

        with open(data_path,'a') as f:
            writer = csv.writer(f)
            writer.writerow([train_loss, val_loss, train_acc, val_acc])
        
        if epoch % 10 ==0:
            print('loss = {}      accuracy = {}      val_loss = {}      val_accuracy: {}'.format(train_loss, train_acc, val_loss, val_acc))

    print('Final training loss =   {}      final training accuracy =   {}'.format(train_loss, train_acc))
    print('Final validation loss = {}      final validation accuracy = {}'.format(val_loss, val_acc))

    # Save state dict for future loading of trained model
    torch.save(model.state_dict(), f'data/model_chk/artist_noise_classifier_epoch{epoch+1}.pt')
    torch.save(opt.state_dict(), f'data/model_chk/artist_noise_classifier_opt_epoch{epoch+1}.pt')
    print('Saved state dict')

    return train_losses, validation_losses, train_accs, validation_accs