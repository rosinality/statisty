class RoseTable:
    def __init__(self):
        self.char_horizontal = '-'
        self.char_vertical = '|'
        self.char_top = '='
        self.char_bottom = '='
        self.char_header = '='                

        self.char_left_top = '+'
        self.char_left_middle = '+'
        self.char_left_bottom = '+'

        self.char_right_top = '+'
        self.char_right_middle = '+'
        self.char_right_bottom = '+'

        self.char_center_top = '+'
        self.char_center_middle = '+'
        self.char_center_bottom = '+'

        self.padding = 2
        self.left_header = 0

        self.title = ''

        self.table = []
        self.header_id = []
        self.vertical = {}
        self.row_number = []
    
    def add_header(self, header, align = None):
        item = {}
        item['header'] = True
        item['row'] = header
        item['align'] = align
        
        self.row_number.append(len(header))

        self.table.append(item)
        self.header_id.append(len(self.table) - 1)
        
    def add_row(self, row, align = None, *kwarg):
        item = {}
        item['row'] = row
        
        self.table.append(item)

    def add_title(self, title):
        item = {}
        item['title'] = title
        self.table.append(item)

    def add_separator(self, char = '='):
        item = {}
        item['separator'] = char

        self.table.append(item)

    def add_vertical(self, index, char = '|'):
        self.vertical[index] = char
        
    def __repr__(self):
        result = []

        widths = []
        subset = 0
        total = 0
        for index, line in enumerate(self.table):
            if not 'row' in line:
                continue

            if index in self.header_id:
                width = [0] * self.row_number[subset]
                widths.append(width)
                subset += 1

            for no, row in enumerate(line['row']):
                len_str = len(str(row)) + self.padding
                if len_str > width[no]:
                    width[no] = len_str

            total = max(total, sum(width))

        for no, width in enumerate(widths):
            add = (total - sum(width)) // self.row_number[no]
            if add < 0:
                continue
            widths[no] = list(map(lambda x: x + add, width))
        
        if self.title:
            length = total - len(self.title)

            if length % 2 == 1:
                length += 1
                total += 1

            result.append(' ' * (length // 2) + self.title)

        total -= 2

        if self.char_top:
            result.append(self.char_top * total)

        for no, index in enumerate(self.header_id):
            try:
                table = self.table[index:self.header_id[no + 1]]

            except IndexError:
                table = self.table[index:]

            row_number = self.row_number[no]
            width = widths[no]

        #   for no in self.vertical.keys():
        #       width[no - 1] += self.padding + 1
            align_format = {'l': '{:<', 'c': '{:^', 'r': '{:>'}

            for item in table:
                try:
                    result.append(item['separator'] * total)
                    continue
                except KeyError:
                    pass
                try:
                    length = total - len(item['title'])

                    if length % 2 == 1:
                        length += 1
                        total += 1

                    result.append(' ' * (length // 2) + item['title'])
                    continue
                except KeyError:
                    pass
                try:
                    align = item['align']
                except KeyError:
                    pass

                row = item['row']
                temp = []
                for no, cell in enumerate(row):
                #   if no + 1 in self.vertical:
		    #       this_width = width[no] - 1
		    #   else:
                    this_width = width[no]
		    
                    try:
                        formatting = align_format[align[no]]
                    except:
                        formatting = align_format['l']

                    temp.append((
                        formatting + str(this_width - self.padding) +
                        '}' + ' ' * self.padding).format(cell))
		    #   if no + 1 in self.vertical:
		    #       temp.append(self.vertical[no + 1] + ' ' * self.padding)
		    
                

                line = ''.join(temp)
                if line.strip():
                    result.append(''.join(temp))

                    if 'header' in item and self.char_header:
                        result.append(self.char_header * total)

            if self.char_bottom:
                result.append(self.char_bottom * total)

        return '\n'.join(result)
