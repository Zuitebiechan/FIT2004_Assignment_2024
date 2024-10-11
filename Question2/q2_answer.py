# ======================================================== Q2 Code =================================================================

# ----------------------------------------------- Class for TrieNode ----------------------------------------------------
class TrieNode:
    """ 
    The TrieNode object for Trie class.

    Attributes:
        link:               A list which stores the connected child Nodes. 
                            The index of the link list represents the next alphabet or digit character.
                                
        frequency:          An integer value which represents the frequency of most frequent completed sentence,  
                            this sentence has prefix up until current node. (this sentence begins with string value up until current node)

        char:               A character value which represents the character of the current node.

        data:               Terminal node       A string value, represents the completed sentence in the terminal node.
                            Other nodes         None 
                            
        words_list:         A list of tuples, where each tuple contains a word and its frequency, stores all the words in the subtree of the node. 
    """
    def __init__(self, size=63) -> None:
        """
        Description: 
            Constructor for TrieNode object.
        Input:
            size:   int value for the size of link list, it is the number of unique possible char in sentence.
                    default value is 63 for 26 lower case alphabets, 26 upper case alphabets, 10 digits and $ (terminal node).
        Time complexity: 
            Best and Worst: O(1)
        Aux space complexity: 
            In-place, O(1)
        """
        self.link = [None] * size  # 0: $, 1-26: a-z, 27-52: A-Z, 53-62: 0-9
        self.frequency = 0
        self.char = None
        self.data = None
        self.words_list = []  # List to store tuples of (word, frequency)
        
    def get_index(self, char: str) -> int:
        """
        Description:
            Get the index of the character in the link list.
        Input:
            char:       A character value which represents the character of the current node.
        Output:
            An integer value representing the index of the character in the link list.
        Time complexity:
            Best and Worst: O(1)
        Aux space complexity:
            In-place, O(1)
        """
        if char == '$':
            return 0
        elif 'a' <= char <= 'z':
            return ord(char) - ord('a') + 1       # Indices 1-26
        elif 'A' <= char <= 'Z':
            return ord(char) - ord('A') + 27      # Indices 27-52
        elif '0' <= char <= '9':
            return ord(char) - ord('0') + 53      # Indices 53-62
        else:
            return -1  # Invalid character
        
# ----------------------------------------------- Class for Trie ----------------------------------------------------
class Trie:
    """
    The Trie object for storing the sentences and their frequencies.

    Attributes:
        root:                       A TrieNode object, represent the root node of the Trie.
    """
    def __init__(self) -> None:
        """
        Description:
            Constructor for Trie object.
        Time complexity:
            Best and Worst: O(1)
        Aux space complexity:
            In-place, O(1)
        """
        self.root = TrieNode()
    
    def insert_recursion(self, current: TrieNode, word: str, i=0) -> tuple:
        """
        Description:
            Insert a word into the Trie recursively.
        Input:
            current:                A TrieNode object representing the current node in the Trie.
            word:                   A string value representing the word to be inserted.
            i:                      An integer value representing the index of the character in the word.
                                    Default value is 0.
        Output:
            A tuple containing the index of the character in the link list, the frequency of the word, and the word.
            [ index, frequency, word ]

        Exceptions:
            ValueError:             Raised when the character is out of the range of valid characters.

        Time complexity:
            Best and Worst: O(M) where M is the length of the word.
        Analysis:
            The method processes each character of the word exactly once as it traverses or inserts into the Trie.

        Aux space complexity:
            O(M) where M is the length of the word.
        Analysis:
            The method uses the stack space for the recursive calls which is proportional to the length of the word.
        """
        # Base case: if we've reached the end of the word, handle the terminal node
        if i == len(word):
            index = 0  # The terminal node is always stored at index 0 in the link list
            
            # If the terminal node does not exist, create it and set its properties
            if current.link[index] is None:
                current.link[index] = TrieNode()
                current.link[index].char = '$'  # Use '$' to represent the end of a word
                current.link[index].data = word  # Store the word itself in the terminal node
            
            # Move to the terminal node and increment its frequency count
            current = current.link[index]
            current.frequency += 1
            
            # Return the index, frequency of the word, and the word itself
            return index, current.frequency, current.data
        else:
            # Recursive case: process the next character in the word
            char = word[i]  # Get the character at the current position
            index = current.get_index(char)  # Compute the index in the link list for this character
            
            # If the character is invalid, raise an error
            if index == -1:
                raise ValueError("Invalid character in the input word")

            # If the link for this character does not exist, create a new Trie node
            if current.link[index] is None:
                current.link[index] = TrieNode()
                current.link[index].char = char  # Set the character in the newly created node
            
            # Move to the next node in the Trie (corresponding to the character)
            current = current.link[index]
            
            # Recursively call the function for the next character in the word
            self.insert_recursion(current, word, i + 1)
            
            # Update the intermediate node's words_list with terminal and child words
            self._update_words_list(current, word) # O(1)
            
            # Return the index of the character, its frequency, and the word itself
            return index, current.frequency, current.data
        
    def _update_words_list(self, curr_node: TrieNode, word: str) -> None:
        """    
        Description:
            Update the words_list of the node to include the given word in the correct position,
            maintaining a fixed-size list sorted by frequency (descending) and lexicographically.
            This method ensures that the words_list of the node is always sorted by word frequency in descending order.
            If multiple words have the same frequency, they are sorted lexicographically (alphabetically). The list has
            a fixed size of 10 to limit memory usage, keeping only the most relevant words for the node.

        Input:
            curr_node:          The TrieNode object whose words_list is to be updated.
            word:               A string represents the word to be added or updated in the node's words_list.

        Time Complexity:
            Best & Worst: O(1) in practice due to the fixed maximum list size, with each operation involving a limited number of comparisons.

        Auxiliary Space Complexity:
            O(1) since the words_list size is capped at a fixed maximum of 10 entries.
        """
        max_list_size = 10  # Limit the size of the list to the top 10 most frequent words

        # Check if the word already exists in the words_list
        for i, (existing_word, frequency) in enumerate(curr_node.words_list):
            if existing_word == word:
                # If the word is found, update its frequency
                curr_node.words_list[i] = (existing_word, frequency + 1)

                # Move the updated word to its correct position using insertion-based sorting by frequency and lexicographical order
                while i > 0 and curr_node.words_list[i][1] > curr_node.words_list[i - 1][1]:
                    curr_node.words_list[i], curr_node.words_list[i - 1] = curr_node.words_list[i - 1], curr_node.words_list[i]
                    i -= 1
                while i > 0 and curr_node.words_list[i][1] == curr_node.words_list[i - 1][1] and curr_node.words_list[i][0] < curr_node.words_list[i - 1][0]:
                    curr_node.words_list[i], curr_node.words_list[i - 1] = curr_node.words_list[i - 1], curr_node.words_list[i]
                    i -= 1
                return  # Exit the function once the word has been updated and repositioned

        # If the word is not in the list, insert it in the correct position
        new_entry = (word, 1)

        # If the list is not full, directly append the new entry in sorted order
        if len(curr_node.words_list) < max_list_size:
            curr_node.words_list.append(new_entry)
            # Insert the new entry in the correct position using insertion-based sorting by frequency and lexicographical order
            i = len(curr_node.words_list) - 1
            while i > 0 and curr_node.words_list[i][1] > curr_node.words_list[i - 1][1]:
                curr_node.words_list[i], curr_node.words_list[i - 1] = curr_node.words_list[i - 1], curr_node.words_list[i]
                i -= 1
            while i > 0 and curr_node.words_list[i][1] == curr_node.words_list[i - 1][1] and curr_node.words_list[i][0] < curr_node.words_list[i - 1][0]:
                curr_node.words_list[i], curr_node.words_list[i - 1] = curr_node.words_list[i - 1], curr_node.words_list[i]
                i -= 1
        else:
            # If the list is full, check if the new entry should replace the lowest-ranked word
            lowest_word, lowest_frequency = curr_node.words_list[-1]
            if new_entry[1] > lowest_frequency or (new_entry[1] == lowest_frequency and new_entry[0] < lowest_word):
                curr_node.words_list[-1] = new_entry  # Replace the lowest-ranked word
                # Adjust the position of the replaced word using insertion-based sorting by frequency and lexicographical order
                i = len(curr_node.words_list) - 1
                while i > 0 and curr_node.words_list[i][1] > curr_node.words_list[i - 1][1]:
                    curr_node.words_list[i], curr_node.words_list[i - 1] = curr_node.words_list[i - 1], curr_node.words_list[i]
                    i -= 1
                while i > 0 and curr_node.words_list[i][1] == curr_node.words_list[i - 1][1] and curr_node.words_list[i][0] < curr_node.words_list[i - 1][0]:
                    curr_node.words_list[i], curr_node.words_list[i - 1] = curr_node.words_list[i - 1], curr_node.words_list[i]
                    i -= 1

    def check(self, input_word: str, num_suggestions=3) -> list:
        """
        Description:
            Check the Trie for the input word and return a list of suggestions.

        Input:
            input_word:             A string representing the word to check in the Trie.
            num_suggestions:        An integer representing the maximum number of suggestions to return (default is 3).

        Output:
            A list of strings representing the top suggestions for the input word, based on frequency and longest prefix match.

        Time Complexity:
            Best: O(M) where M is the length of the input word.
            Worst: O(M + U) where M is the length of the input word and U is the total number of characters in the words returned.
        Analysis:
            Best: The best case occurs when the exact word exists in the Trie, meaning the method does not need to collect any suggestions.
                •	The Trie traversal checks each character of the word, which takes O(M) time.
	            •	Since the exact word is found, the method returns an empty list immediately, without processing any further nodes or suggestions.
            Worst: When the input word is partially present in the Trie, the method proceeds to collect suggestions based on the longest matching prefix.
                •   In the worst-case scenario, the method will traverse all M characters of the input word to check if it exists in the Trie. This traversal takes linear time O(M).
                •   The method then recursively collects suggestions based on the longest matching prefix, which involves traversing the Trie and collecting suggestions. 
                    The total time complexity for this operation is O(U), where U is the total number of characters in the collected words.

        Auxiliary Space Complexity:
            O(1) for internal operations since it uses a fixed-size list for suggestions.
        """
        # First, check if the exact word exists in the Trie
        if self._is_exact_match(self.root, input_word): # O(L)
            return []  # If the exact match is found, return an empty list

        solution = []  # List to store collected words and their frequencies
        # Recursively collect suggestions based on the longest matching prefix
        self._collect_suggestions_recursive(self.root, input_word, 0, solution) # O(L + U)

        # Return at most num_suggestions words from the solution, based on the provided limit
        # Create the result list iteratively without using slices
        result = []
        for word in solution: # O(U)
            result.append(word)
            if len(result) == num_suggestions:
                break  # Stop once we have collected the desired number of suggestions

        return result

    def _is_exact_match(self, current: TrieNode, word: str) -> bool:
        """
        Description:
            Check if the exact word exists in the Trie.

        Input:
            current:            The current node in the Trie being traversed.
            word:               A string representing the word to check for an exact match.

        Output:
            Boolean value indicating if the exact word exists in the Trie.

        Time Complexity:
            Best & Worst: O(M) where M is the length of the input word.
        Analysis:
            The method traverses the Trie based on the characters of the input word to check if the exact word exists.
        Auxiliary Space Complexity:
            In-place, O(1)
        """
        for char in word:
            index = current.get_index(char)  # Get the index of the character in the link array
            if index == -1 or current.link[index] is None:
                return False  # If character not found, the word does not exist in the Trie
            current = current.link[index]  # Move to the next node in the Trie

        # Check if we have reached the terminal node for this word
        return current.link[0] is not None and current.link[0].data == word

    def _collect_suggestions_recursive(self, current: TrieNode, word: str, index: int, solution: list) -> None:
        """
        Description:
            Recursively collect suggestions from the Trie based on the longest matching prefix.

        Input:
            current:                The current node in the Trie being traversed.
            word:                   A string representing the word to collect suggestions for.
            index:                  An integer representing the current index of the character in the word being processed.
            solution:               A list to store the collected words and their frequencies.

        Output:
            None (modifies the solution list in place).

        Time Complexity:
            O(M + U) where M is the length of the word and U is the total number of characters in the collected words.
        Analysis:
            The method traverses the Trie based on the longest matching prefix and collects suggestions from the words_list.
            The time complexity is proportional to the length of the input word and the total number of characters in the collected words.
        Auxiliary Space Complexity:
            O(1) for the internal state of each recursive call.
        """
        # Base case: if we've reached the end of the word or the current node is None
        if index == len(word) or current is None:
            # Collect words from the longest matching node's words_list
            if current:
                self._collect_suggestions(current, solution) # O(U)
            return

        char = word[index]  # Get the character at the current index
        char_index = current.get_index(char)  # Compute the index in the link list for this character

        # If the character exists, continue recursively to the next node
        if current.link[char_index] is not None:
            next_node = current.link[char_index]
            self._collect_suggestions_recursive(next_node, word, index + 1, solution) 

        # Collect suggestions from the current node's words_list, including intermediate nodes
        if current and current.words_list:
            self._collect_suggestions(current, solution)

    def _collect_suggestions(self, node: TrieNode, solution: list) -> None:
        """
        Description:
            Collect data from the node's words_list and add them to the solution list.

        Input:
            node:           The TrieNode object from which to collect suggestions.
            solution:       A list to store the collected words and their frequencies.

        Output:
            None (modifies the solution list in place).

        Time Complexity:
            Best & Worst: O(U) where U is the total number of characters in the words in node.words_list.

        Auxiliary Space Complexity:
            In-place: O(1)
        """
        if node and node.words_list:
            # Iterate through each word in the words_list and add it to the solution if not already present
            for word_data, frequency in node.words_list:
                if word_data not in solution:
                    solution.append(word_data)

    def _print_trie_recursive(self, node, level=0):
        indent = '    ' * level
        if level == 0:
            print(f"{indent}Root (frequency: {node.frequency}, data: {node.data})")
        else:
            print(f"{indent}Node '{node.char}' (frequency: {node.frequency}, data: {node.data}, words_list: {node.words_list})")
        for idx, child in enumerate(node.link):
            if child:
                self._print_trie_recursive(child, level + 1)

# ----------------------------------------------- Class for SpellChecker ----------------------------------------------------
class SpellChecker:
    """
    The SpellChecker object for checking the spelling of words using a Trie.

    Attributes:
        trie:                       A Trie object to store the words and their frequencies.
    """
    def __init__(self, filename: str) -> None:
        """
        Description:
            Constructor for SpellChecker object.

        Input:
            filename:               A string representing the name of the file containing the dictionary of words.

        Time Complexity:
            Best and Worst: O(T) where T is the total number of characters in the file.
        Analysis:
            The method reads each character in the file to build the Trie.

        Auxiliary Space Complexity:
            O(N) where N is the total number of characters in the file.
        Analysis:
            The method uses the Trie to store the words and their frequencies.
        """ 
        self.trie = Trie()
        self.build_trie(filename)
    
    def build_trie(self, filename: str) -> None:
        """
        Description:
            Build the Trie using the words from the input file.

        Input:
            filename:            A string representing the name of the file containing the dictionary of words.

        Time Complexity:
            Best and Worst: O(T) where T is the total number of characters in the file.
        Analysis:
            The method reads each character in the file to build the Trie.

        Auxiliary Space Complexity:
            O(N) where N is the total number of characters in the file.
        Analysis:
            The method uses the Trie to store the words and their frequencies.
        """
        with open(filename, 'r') as file:
            for line in file:
                words = re.findall(r'[A-Za-z0-9]+', line)
                for word in words: 
                    self.trie.insert_recursion(self.trie.root, word) # O(L)
    
    def check(self, input_word: str) -> list:
        """
        Description:
            Check the spelling of the input word and return a list of suggestions.

        Input:
            input_word:          A string representing the word to check the spelling.
        Output:
            A list of strings representing the top suggestions for the input word.

        Time Complexity:
            Best: O(M) where M is the length of the input word.
            Worst: O(M + U) where M is the length of the input word and U is the total number of characters in the words returned.
        Analysis:
            Best: The best case occurs when the exact word exists in the Trie, meaning the method does not need to collect any suggestions.
            Worst: When the input word is partially present in the Trie, the method proceeds to collect suggestions based on the longest matching prefix.

        Auxiliary Space Complexity:
            O(1) for internal operations since it uses a fixed-size list for suggestions.
        Analysis:
            The method calls the Trie's check method to get the suggestions.
        """
        return self.trie.check(input_word) # O(M + U)
    
    def print_trie(self):
        self.trie._print_trie_recursive(self.trie.root)

# # # Example usage
# myChecker = SpellChecker("Messages.txt")

# print(myChecker.check("IDK"))   # Expected Output：[]
# print(myChecker.check("zoo"))   # Expected Output：[]
# print(myChecker.check("LOK"))   # Expected Output：["LOL", "LMK"]
# print(myChecker.check("IDP"))   # Expected Output：["IDK", "IDC", "I"]
# print(myChecker.check("Ifc"))   # Expected Output：["If", "I", "IDK"]

# Print the Trie structure
# myChecker.print_trie()
