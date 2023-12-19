# Containers for the VHR Cloud Masking Software

## Container Infrastructure

This repository includes two containers, vhr-cloudmask:latest (production) and vhr-cloudmask:dev (development).
The production container includes all the dependencies needed to run the cloud masking software.
The development container has the main dependencies with the exception of tensorflow-caney and the 
vhr-cloudmask pip package. This to add the development of software without having to install these packages.

All containers are built on a weekly schedule for patching and to keep dependencies up to date.
New releases will trigger the build of the production container.

## Building Singularity Sandbox

In some systems you will need to build a Singularity sandbox in order to use a container, particularly
on systems where setuid cannot be set.

```bash
singularity build --sandbox /lscratch/$USER/container/tensorflow-caney docker://nasanccs/vhr-cloudmask:latest
```

## Downloading as a Singularity container

If you only need to dowload the container, you can simply do so by typing the singularity pull command.

```bash
singularity pull docker://nasanccs/vhr-cloudmask:latest
```

## Downloading as a Singularity container (Development)

If you only need to dowload the container, you can simply do so by typing the singularity pull command.

```bash
singularity pull docker://nasanccs/vhr-cloudmask:dev
```