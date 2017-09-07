#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make

 
CORPUS=tok_hpi_corpus_glove
VOCAB_FILE=vocab_my.txt
COOCCURRENCE_FILE=cooccurrence_my.bin
COOCCURRENCE_SHUF_FILE=cooccurrence_my.shuf.bin
BUILDDIR=build
MYDIR=mycorpus
SAVE_FILE=vectors_my
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=1
VECTOR_SIZE=300
MAX_ITER=15
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=8
X_MAX=10

echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $MYDIR/$VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $MYDIR/$VOCAB_FILE
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $MYDIR/$VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $MYDIR/$COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $MYDIR/$VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $MYDIR/$COOCCURRENCE_FILE
echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $MYDIR/$COOCCURRENCE_FILE > $MYDIR/$COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $MYDIR/$COOCCURRENCE_FILE > $MYDIR/$COOCCURRENCE_SHUF_FILE
echo "$ $BUILDDIR/glove -save-file $MYDIR/$SAVE_FILE -threads $NUM_THREADS -input-file $MYDIR/$COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $MYDIR/$VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $MYDIR/$SAVE_FILE -threads $NUM_THREADS -input-file $MYDIR/$COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $MYDIR/$VOCAB_FILE -verbose $VERBOSE
