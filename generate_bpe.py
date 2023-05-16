import os
import codecs
import argparse
import subword_nmt.learn_bpe
import subword_nmt.apply_bpe

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True,
                    help='Path to input file')
parser.add_argument('--output', type=str, required=True,
                    help='Path to output file')
parser.add_argument('--num-merge', type=int, default=10000,
                    help='Number of BPE merges (default: 10000)')
args = parser.parse_args()

# Read input file
with codecs.open(args.input, 'r', 'utf-8') as f:
    lines = f.readlines()

# Split each line into constituent tokens
sentences = []
for line in lines:
    tokens = line.strip().split()
    sentence = ' '.join([token.split('|')[0] for token in tokens])
    sentences.append(sentence)

# Learn BPE
output_file = r"C:\Users\HP\Downloads\Findings-of-EMNLP-2020-Code-Mixed\Findings-of-EMNLP-2020-Code-Mixed\hi\cm\bpe_vocab.txt"
with open(args.output, 'w', encoding='utf-8') as outfile:
    bpe_model = subword_nmt.learn_bpe.learn_bpe(sentences, num_symbols=args.num_merge,outfile=outfile )
    # outfile.write('#version: 0.2\n')



# Load BPE encoder
with open(args.output, 'r+', encoding='utf-8') as outfile:

    bpe_encoder = subword_nmt.apply_bpe.BPE(outfile)


# Encode input lines using BPE encoder
encoded_lines = []
for line in lines:
    tokens = line.strip().split()
    sentence = ' '.join([token.split('|')[0] for token in tokens])
    encoded_sentence = bpe_encoder.segment(sentence).strip()
    encoded_lines.append(encoded_sentence)

# Do something with encoded lines...


# python generate_bpe.py --input input.txt --output bpe.vocab --num-merge 10000

