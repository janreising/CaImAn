from pushbullet import Pushbullet


class Comm:

    def __init__(self, api_key=None):

        if api_key is None:
            api_key = "o.exf0kTX0aUW8Y4hd6qNo1jDe99HaOF34"

        self.pb = Pushbullet(api_key)

    def push_text(self, title, msg):
        push = self.pb.push_note(title, msg)

