from prefixspan import PrefixSpan
class PatternFinder:
    def __init__(self, raw_data, labels, n_classes=10):
        """Initialize the PatternFinder class."""
        self.raw_data = raw_data
        self.labels = labels
        self.n_classes = n_classes
        self.patterns_by_class = {}
        self.common_patterns = set()

    def preprocess_data(self):
        """Preprocess data to retain 'PageView' events and extract relevant information."""
        processed_data = []
        for label in range(self.n_classes):
            # Filter data belonging to the current class
            data_for_label = self.raw_data[self.labels == label]
            data_for_label = data_for_label.tolist()
            # Keep only events of type 'PageView' in each sequence
            for j in range(len(data_for_label)):
                data_for_label[j] = [i for i in data_for_label[j] if i[2] == 'PageView']
            # Extract the relevant property of events
            for j in range(len(data_for_label)):
                data_for_label[j] = [i[3] for i in data_for_label[j]]
            processed_data.append(data_for_label)
        return processed_data

    def find_patterns(self):
        """Use the PrefixSpan algorithm to find frequent patterns for each class."""
        processed_data = self.preprocess_data()
        for label, data in enumerate(processed_data):
            ps = PrefixSpan(data)
            min_support = len(data) // 3  # Define minimum support
            # Store patterns for this class, excluding those that meet the min_support
            self.patterns_by_class[label] = set(tuple(pattern[1]) for pattern in ps.frequent(min_support))

        # Identify common patterns across all classes
        self.common_patterns = set.intersection(*self.patterns_by_class.values())

        # Remove these common patterns from the pattern set of each class
        for label in self.patterns_by_class:
            self.patterns_by_class[label] -= self.common_patterns

    def get_patterns_by_class(self):
        """Return unique patterns in each class, after removing common patterns."""
        return self.patterns_by_class

    def get_common_patterns(self):
        """Return patterns that are common across all classes."""
        return self.common_patterns
