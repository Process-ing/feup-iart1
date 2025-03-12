from abc import abstractmethod
from view.viewer.pygameviewer import PygameViewer


class ButtonViewer(PygameViewer[None]):
    @abstractmethod
    def is_clicked(self) -> bool:
        pass
