import numpy as np
import re
from test_model import encoder_model, decoder_model, num_decoder_tokens, num_encoder_tokens, input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, max_encoder_seq_length

class ChatBot:
    def __init__ (self):
        self.negative_commands = ["no", "nope", "nah", "not a chance", "no thanks", "never", "nuh uh"]
        self.exit_commands = ["quit", "pause", "exit", "bye", "later", "stop"]

    def start_chat(self):
        user_response = input("Hi, I am a chatbot trained on twitter tweets on the weather. Would you like to chat with me today? ")

        if user_response in self.negative_commands:
            print("Ok, have a nice day!")
            return
        
        self.chat(user_response)

    def chat(self, reply):
        while not self.make_exit(reply):
            reply = input(self.generate_responses(reply))

    def string_to_matrix(self, user_input):
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix
        
    def generate_responses(self, user_input):
        input_matrix = self.string_to_matrix(user_input)
        states_value = encoder_model.predict(input_matrix)
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        target_seq[0, 0, target_features_dict['<START>']] = 1.

        chatbot_response = ''

        stop_condition = False
        while not stop_condition:
            output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
      
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_features_dict[sampled_token_index]
      
            chatbot_response += " " + sampled_token
      
            if (sampled_token == '<END>' or len(chatbot_response) > max_decoder_seq_length):
                stop_condition = True
        
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
      
            states_value = [hidden_state, cell_state]

        chatbot_response = chatbot_response.replace("<START>","").replace("<END>","")

        return chatbot_response
    
    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print("Ok, have a good day!")
                return True
        return False
        

weather_bot = ChatBot()

weather_bot.start_chat()
    
    
