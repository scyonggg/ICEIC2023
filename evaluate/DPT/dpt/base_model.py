import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
#                self.model_dict = self.posenet.state_dict()
#                self.pretrained_dict = torch.load(self.posenet_path)
#                self.pretrained_dict = {k: v for k, v in self.pretrained_dict.items() if k in self.model_dict}
#                self.model_dict.update(self.pretrained_dict) 
#                self.posenet.load_state_dict(self.model_dict)
        if True:
            model_dict = self.state_dict()
            pretrained_dict = torch.load(path , map_location=torch.device("cpu"))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            self.load_state_dict(model_dict)


        if False:
            parameters = torch.load(path, map_location=torch.device("cpu"))
    
            if "optimizer" in parameters:
                parameters = parameters["model"]

            self.load_state_dict(parameters)
