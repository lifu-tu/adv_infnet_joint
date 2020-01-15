import sys

def process(filename):
     f = open(filename, 'r')
     data = f.readlines()
     f.close()
     #f = open(filename + '_devacc', 'w')
     result = []
     for line in data:
          """
          if ('inf' in line):
                a = line.index('inf')
                result.append(line[a+4: a+11])
          """
          if 'gradient norm:' in line:
                a = line.index('gradient norm:')
                result.append(line[a+14:a+20])
     print ','.join(result[:20])

if __name__ == '__main__':
     process(sys.argv[1])


