TARGET 		= openmp

CC 			= gcc

CFLAGS 		= -W -Wall -Ofast -fopenmp

LINKER 		= gcc -o 

LFLAGS 		= -Wall -I. -lm -lGL -lGLEW -lSDL2 -lGLU  -lSDL2main -fopenmp

SRCDIR 		= src
OBJDIR 		= obj
BINDIR 		= bin

SOURCES 	:= $(wildcard $(SRCDIR)/*.c)
INCLUDES 	:= $(wildcard $(SRCDIR)/*.h)
OBJECTS 	:= $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
rm 			= rm -f

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(OBJECTS)
	@$(LINKER) $@ $(OBJECTS) $(LFLAGS)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.c
	@$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	@$(rm) $(OBJECTS)

.PHONY: remove
remove: clean
	@$(rm) $(BINDIR)/$(TARGET)

