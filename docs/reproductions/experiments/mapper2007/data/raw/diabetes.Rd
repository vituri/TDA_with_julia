\name{diabetes}
\alias{diabetes}
\docType{data}

\title{Reaven and Miller diabetes data}

\description{
The data set contains five measurements made on 145 non-obese adult patients classified into three groups.

The three primary variables are glucose intolerance (area under the straight
line connecting glucose levels), insulin response to oral glucose (area under
the straight line connecting insulin levels) and insulin resistance (measured
by the steady state plasma glucose (SSPG) determined after chemical suppression
of endogenous insulin secretion). Two additional variables, the relative weight
and fasting plasma glucose, are also included.

Reaven and Miller, following Friedman and Rubin (1967), applied cluster analysis
to the three primary variables and identified three clusters: "normal",
"chemical diabetic", and "overt diabetic" subjects. The column \code{group}
contains the classifications of the subjects into these three groups,
obtained by current medical criteria.
}

\usage{
data(diabetes)
}

\format{A data frame with the following variables:
\describe{
\item{rw}{relative weight, expressed as the ratio of actual weight to expected weight, given the person's height.}
\item{fpg}{fasting plasma glucose level.}
\item{glucose}{area under plasma glucose curve after a three hour oral glucose tolerance test (OGTT).}
\item{insulin}{area under plasma insulin curve after a three hour oral glucose tolerance test (OGTT).}
\item{sspg}{Steady state plasma glucose, a measure of insulin resistance.}
\item{group}{the type of diabetes: a factor with levels \code{normal}, \code{chemical} and \code{overt}.}}
}
\source{
Reaven, G. M. and Miller, R. G. (1979). An attempt to define the nature of chemical diabetes using a multidimensional analysis. \emph{Diabetologia} 16, 17--24.
Andrews, D. F. and Herzberg, A. M. (1985). \emph{Data: A Collection of Problems from Many Fields for the Student and Research Worker}, Springer-Verlag, Ch. 36.
}

\references{
Reaven, G. M. and Miller, R. G. (1979). An attempt to define the nature of chemical diabetes using a multidimensional analysis. \emph{Diabetologia} 16, 17--24.

Friedman, H. P. and Rubin, J. (1967). On some invariant criteria for grouping
data. \emph{Journal of the American Statistical Association} \bold{62}, 1159--1178.

Hawkins, D. M. and McLachlan, G. J., 1997. High-breakdown linear
discriminant analysis. \emph{Journal of the American Statistical
Association} \bold{92} (437), 136--143.
}

\examples{
data(diabetes)
(cc <- Linda(group~insulin+glucose+sspg, data=diabetes))
(pr <- predict(cc))
}

\keyword{datasets}
