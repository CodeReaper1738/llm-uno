from rlcard.games.uno.game import UnoGame

class CustomUnoGame(UnoGame):
    def get_payoffs(self):
        ''' Return modified payoffs '''
        winner = self.round.winner
        if winner is not None and len(winner) == 1:
            winner_id = winner[0]
            print(f"Player {winner_id} won the game!")
            self.payoffs = [-1] * self.num_players
            self.payoffs[winner_id] = 1
        else:
            print(f"No winner detected. Payoffs: {self.payoffs}")
        return self.payoffs
