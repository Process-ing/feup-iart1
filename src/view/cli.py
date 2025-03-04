from src.model.building import Building

class Cli:
    def __init__(self):
        pass

    @staticmethod
    def __get_tokens(text: str) -> list[str]:
        return text.strip().split()

    def __print_prefix(self) -> None:
        print("[router-solver]# ", end="")

    def read_input(self) -> list[str]:
        self.__print_prefix()

        try:
            text = input()
        except EOFError:
            return ['exit']

        tokens = self.__get_tokens(text)
        return tokens

    def print_building(self, building: Building) -> None:
        print(building)

    def print_error(self, message: str) -> None:
        print(message)